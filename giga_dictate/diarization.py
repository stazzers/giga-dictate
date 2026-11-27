"""Local CAM++ embeddings and session-only clustering of numbered speakers."""

from copy import copy
from dataclasses import replace

import numpy as np

from .meeting_types import Speaker, SpeakerEvidence, SpeechSegment
from .speaker_clustering import (
    MAX_REPRESENTATIVES,
    average_linkage,
    stable_mapping,
    supported_split,
)

MODEL_ID = "iic/speech_campplus_sv_zh_en_16k-common_advanced@v1.0.0"
PREPROCESSING = "kaldi-fbank80-16k-mean-v1"


class SpeakerCountError(ValueError):
    """Controlled count-validation message safe to show without model inputs."""


def unit(vector):
    vector = np.asarray(vector, np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if vector.shape != (192,) or not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError("Некорректный голосовой вектор")
    return vector / norm


def gallery_center(vectors):
    mean = np.mean(vectors, axis=0)
    return unit(mean) if np.linalg.norm(mean) >= 1e-8 else unit(vectors[0])


class CAMPlusEmbedder:
    def __init__(self, path):
        import kaldi_native_fbank as knf
        import onnxruntime as ort
        self.knf = knf
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.fbank_options = self.knf.FbankOptions()
        self.fbank_options.frame_opts.dither = 0
        self.fbank_options.frame_opts.samp_freq = 16000
        self.fbank_options.frame_opts.snip_edges = True
        self.fbank_options.mel_opts.num_bins = 80
        self.fbank_options.mel_opts.debug_mel = False

    def __call__(self, pcm):
        fbank = self.knf.OnlineFbank(self.fbank_options)
        fbank.accept_waveform(16000, np.asarray(pcm, np.float32).tolist())
        fbank.input_finished()
        if fbank.num_frames_ready < 10:
            raise ValueError("Недостаточно речи для голосового вектора")
        features = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        features -= features.mean(axis=0, keepdims=True)
        vector = self.session.run(None, {self.input_name: features[None]})[0]
        return unit(vector)


class OnlineDiarizer:
    # Initial conservative defaults, exposed for corpus-based calibration.
    def __init__(self, embedder, match=.5, new=.3, speaker_offset=0, speaker_count=None,
                 cluster_match=.45):
        self.embedder = embedder
        self.match, self.new = match, new
        self.speaker_offset = speaker_offset
        # Count is for this diarized channel, excluding the fixed online mic.
        self.speaker_count = speaker_count
        self.cluster_match = cluster_match
        self.centers = {}
        self.references = {}
        self.reference_origins = {}
        self.counts = {}
        self.aliases = {}
        self.candidates = {}
        self.candidate_count = 0
        self.refinement_stats = {}
        self.speakers = {"unknown": Speaker("unknown", "Спикер не определён")}

    def identify(self, vector, *, start=None, end=None, seconds=1.0, quality=1.0):
        vector = unit(vector)
        if quality < .65:
            return "unknown"
        if self.centers:
            scores = {key: max(float(center @ vector),
                               max(float(ref @ vector) for ref in self.references[key]))
                      for key, center in self.centers.items()}
            best = max(scores, key=scores.get)
            runner = max((score for key, score in scores.items() if key != best), default=0.)
            if scores[best] >= self.match and scores[best] - runner >= .15:
                count = self.counts[best]
                if seconds >= 1.0:
                    self.centers[best] = unit(self.centers[best] * min(count, 20) + vector)
                    self.counts[best] += 1
                    if scores[best] >= .55 and max(float(ref @ vector) for ref in self.references[best]) < .9:
                        self.references[best] = [self.references[best][0], *self.references[best][-6:], vector.copy()]
                        self.reference_origins[best] = [None] * len(self.references[best])
                return best
            if scores[best] > self.new:
                return "unknown"
            scores = {key: float(unit(np.mean(item["vectors"], axis=0)) @ vector)
                      for key, item in self.candidates.items()}
            candidate = max(scores, key=scores.get) if scores else None
            if candidate is not None and scores[candidate] >= .65:
                runner = max((score for key, score in scores.items() if key != candidate), default=0.)
                if scores[candidate] - runner < .1:
                    return "unknown"
            if candidate is None or scores[candidate] < .65:
                self.candidate_count += 1
                candidate = f"pending-{self.candidate_count}"
                if len(self.candidates) >= 64:
                    del self.candidates[min(self.candidates, key=lambda k: self.candidates[k]["seconds"])]
                self.candidates[candidate] = {"vectors": [], "seconds": 0., "end": -np.inf}
            item = self.candidates[candidate]
            novel = seconds if start is None else seconds * max(0., end - max(start, item["end"])) / max(end - start, 1e-6)
            item["seconds"] += novel
            if end is not None:
                item["end"] = max(item["end"], end)
            if novel > 0:
                item["vectors"] = [*item["vectors"][-7:], vector.copy()]
            if item["seconds"] < 2.4 - 1e-6:
                return candidate
            vector = unit(np.mean(item["vectors"], axis=0))
            refs = item["vectors"]
        else:
            candidate, refs = None, [vector]
            if seconds < 1.0:
                return "unknown"
        if len(self.centers) >= 15:
            return "unknown"
        number = max([self.speaker_offset, *[int(k.split("-")[1]) for k in self.speakers
                                            if k.startswith("speaker-")]]) + 1
        key = f"speaker-{number}"
        self.centers[key] = vector.copy()
        self.references[key] = [v.copy() for v in refs]
        self.reference_origins[key] = [None] * len(refs)
        self.counts[key] = 1
        self.speakers[key] = Speaker(key, f"Спикер {number}")
        if candidate:
            self.aliases[candidate] = key
            del self.candidates[candidate]
        return key

    def resolve_speaker(self, key):
        visited = set()
        while key in self.aliases and key not in visited:
            visited.add(key)
            key = self.aliases[key]
        return "unknown" if key.startswith("pending-") else key

    def refine(self, turns, evidence):
        """Compute on private state; commit only a complete, validated result."""
        staged = copy(self)
        for attr in ("centers", "references", "reference_origins", "counts", "aliases", "speakers", "refinement_stats"):
            setattr(staged, attr, getattr(self, attr).copy())
        staged_turns = [replace(turn) for turn in turns]
        staged._cluster_meeting(staged_turns, evidence)
        clean_evidence = {}
        for key, item in evidence.items():
            good = np.ones(len(item.vectors), bool) if item.quality is None else np.asarray(item.quality) >= .65
            if good.any():
                clean_evidence[key] = replace(item, vectors=item.vectors[good], weights=item.weights[good])
        staged._refine_known(staged_turns, clean_evidence)
        if self.speaker_count is not None:
            found = {turn.effective_speaker_id for turn in staged_turns
                     if turn.effective_speaker_id != "unknown"
                     and (not self.speaker_offset or turn.source == "system")}
            if len(found) != self.speaker_count:
                raise SpeakerCountError("Недостаточно независимой речи для указанного количества голосов; сохранены предварительные назначения")
        changed = sum(a.speaker_id != b.speaker_id for a, b in zip(turns, staged_turns))
        for attr in ("centers", "references", "reference_origins", "counts", "aliases", "speakers", "refinement_stats"):
            setattr(self, attr, getattr(staged, attr))
        for turn, refined in zip(turns, staged_turns):
            turn.speaker_id, turn.attribution = refined.speaker_id, refined.attribution
        return changed

    def _cluster_meeting(self, turns, evidence):
        """Cluster clean turn summaries, retaining rare voices in a bounded pool."""
        representatives = np.empty((MAX_REPRESENTATIVES, 192), dtype=np.float32)
        anchors, masses, metadata = [], [], []
        ordered = sorted(turns, key=lambda turn: (turn.start, turn.id))
        old_keys = list(self.centers)
        manual = {t.manual_speaker_id for t in turns if t.manual_speaker_id is not None}

        def add(vector, weight, anchor=None, turn=None):
            # Compress only nearly identical observations. Confirmed galleries
            # stay separate until linkage, allowing duplicate speakers to merge.
            count = len(masses)
            if turn is not None and count:
                similarity = representatives[:count] @ vector
                choices = [i for i, key in enumerate(anchors) if key is None]
                best = max(choices, key=lambda i: similarity[i]) if choices else None
                if best is not None and similarity[best] >= .92:
                    representatives[best] = unit(representatives[best] * masses[best] + vector * weight)
                    masses[best] += weight
                    meta = metadata[best]
                    meta["seconds"] += weight
                    meta["turns"] += 1
                    meta["turn_id"] = None
                    meta["longest"] = max(meta["longest"], weight)
                    key = self.resolve_speaker(turn.speaker_id)
                    meta["votes"][key] = meta["votes"].get(key, 0.) + weight
                    return
            if count == MAX_REPRESENTATIVES:
                # Prefer repeated voices over isolated outliers; don't evict
                # gallery anchors. All original summaries remain for assignment.
                choices = [i for i, key in enumerate(anchors) if key is None]
                victim = min(choices, key=lambda i: (metadata[i]["turns"], masses[i]))
                if metadata[victim]["turns"] > 1:
                    # At saturation compact the most redundant pair instead of
                    # permanently refusing a late, acoustically novel voice.
                    matrix = representatives[choices]
                    similarity = matrix @ matrix.T
                    np.fill_diagonal(similarity, -np.inf)
                    a, b = np.unravel_index(int(np.argmax(similarity)), similarity.shape)
                    keep, victim = choices[a], choices[b]
                    representatives[keep] = unit(representatives[keep] * masses[keep]
                                                 + representatives[victim] * masses[victim])
                    masses[keep] += masses[victim]
                    target, source = metadata[keep], metadata[victim]
                    for field in ("seconds", "turns"):
                        target[field] += source[field]
                    target["longest"] = max(target["longest"], source["longest"])
                    target["start"] = min(target["start"], source["start"])
                    target["turn_id"] = None
                    for key, value in source["votes"].items():
                        target["votes"][key] = target["votes"].get(key, 0.) + value
                representatives[victim] = vector
                masses[victim] = weight
                metadata[victim] = dict(seconds=weight, turns=1, longest=weight,
                    start=turn.start, turn_id=turn.id, votes={self.resolve_speaker(turn.speaker_id): weight})
                return
            representatives[count] = vector
            anchors.append(anchor)
            masses.append(max(weight, .1))
            metadata.append(dict(seconds=weight if turn else 0., turns=int(turn is not None),
                longest=weight if turn else 0., start=turn.start if turn else np.inf,
                turn_id=turn.id if turn else None,
                votes={self.resolve_speaker(turn.speaker_id): weight} if turn else {anchor: .01}))

        if self.speaker_count is None:
            for key in old_keys:
                for ref in self.references[key]:
                    add(ref, 0., anchor=key)
        observations = {}
        observed_end = {}
        for turn in ordered:
            item = evidence.get(turn.id)
            if item is None or turn.manual_speaker_id is not None:
                continue
            if self.speaker_offset and turn.source != "system":
                continue
            vectors = np.asarray(item.vectors, np.float32)
            weights = np.asarray(item.weights, np.float32)
            quality = np.ones(len(vectors)) if item.quality is None else np.asarray(item.quality)
            if (vectors.ndim != 2 or vectors.shape[1] != 192 or not 1 <= len(vectors) <= 8
                    or weights.shape != (len(vectors),) or quality.shape != weights.shape
                    or not np.isfinite(vectors).all() or not np.isfinite(weights).all()
                    or not np.isfinite(quality).all() or np.any(weights <= 0)
                    or not np.isfinite(item.speech_seconds) or item.speech_seconds < 0):
                raise ValueError("Некорректная акустическая сводка")
            if item.intervals is not None:
                spans = np.asarray(item.intervals)
                if (spans.shape != (len(vectors), 2) or not np.isfinite(spans).all()
                        or np.any(spans[:, 0] < turn.start - 1e-6)
                        or np.any(spans[:, 1] > turn.end + 1e-6)
                        or np.any(spans[:, 1] <= spans[:, 0])):
                    raise ValueError("Некорректные интервалы акустической сводки")
            clean = quality >= .65
            if not clean.any() or item.speech_seconds < .6:
                continue
            selected = np.stack([unit(v) for v in vectors[clean]])
            center = unit(np.average(selected, axis=0, weights=weights[clean]))
            if np.average(selected @ center >= .65, weights=weights[clean]) < .7:
                continue
            seconds = min(item.speech_seconds, turn.end - turn.start)
            observations[turn.id] = center
            # Ambiguous mixtures cannot bootstrap a new person or train a voice.
            scores = sorted((max(float(center @ ref) for ref in self.references[key])
                             for key in old_keys), reverse=True)
            if (self.speaker_count is None and len(scores) > 1 and scores[0] >= self.match
                    and scores[0] - scores[1] < .15):
                continue
            previous_end = observed_end.get(turn.source, -np.inf)
            novel = max(0., turn.end - max(turn.start, previous_end)) / max(turn.end - turn.start, 1e-6)
            observed_end[turn.source] = max(previous_end, turn.end)
            if novel:
                add(center, seconds * novel, turn=turn)
        if self.speaker_count == 0:
            if any(t.source == "system" for t in turns):
                raise SpeakerCountError("Указан один голос, но слышна речь в системном канале; сохранены предварительные назначения")
            return
        if not masses:
            if self.speaker_count:
                raise SpeakerCountError("Недостаточно независимой речи для указанного количества голосов; сохранены предварительные назначения")
            return
        groups = average_linkage(representatives[:len(masses)], masses, threshold=self.cluster_match,
                                 count=self.speaker_count, anchors=anchors)
        self.refinement_stats = {"representatives": len(masses)}
        if self.speaker_count is None and old_keys:
            # Average-link can isolate a phonetic variant even though a trusted
            # reference recognises it clearly. It is not a new person merely
            # because its similarity to the *other* variants is lower.
            anchored = {anchors[i]: group for group in groups for i in group if anchors[i] is not None}
            retained = []
            for group in groups:
                if any(anchors[i] is not None for i in group):
                    retained.append(group)
                    continue
                center = unit(np.average(representatives[group], axis=0,
                                         weights=[masses[i] for i in group]))
                scores = [max(float(center @ ref) for ref in self.references[key]) for key in old_keys]
                best = int(np.argmax(scores))
                runner = max((s for i, s in enumerate(scores) if i != best), default=0.)
                if scores[best] >= .55 and scores[best] - runner >= .15:
                    anchored[old_keys[best]].extend(group)
                else:
                    retained.append(group)
            groups = retained
        def has_support(group):
            seconds = sum(metadata[i]["seconds"] for i in group)
            count = sum(metadata[i]["turns"] for i in group)
            longest = max(metadata[i]["longest"] for i in group)
            known = any(anchors[i] is not None for i in group)
            if self.speaker_count is not None:
                return seconds >= .6 if self.speaker_count == 1 else (
                    seconds >= 2.4 and (count >= 2 or longest >= 2.4))
            return known or (seconds >= 2.4 and count >= 3) or longest >= 2.4

        supported = [group for group in groups if has_support(group)]
        while self.speaker_count is not None and len(supported) < self.speaker_count:
            best = None
            for index, group in enumerate(supported):
                local_vectors = representatives[group]
                local_weights = [masses[i] for i in group]
                split = supported_split(local_vectors, local_weights,
                    lambda indices: has_support([group[i] for i in indices]))
                if split is not None:
                    partition = [[group[i] for i in indices] for indices in split]
                    means = [unit(np.average(representatives[part], axis=0,
                                            weights=[masses[i] for i in part])) for part in partition]
                    separation = 1 - float(means[0] @ means[1])
                    if best is None or separation > best[0]:
                        best = (separation, index, partition)
            if best is None:
                break
            _, index, partition = best
            supported[index:index + 1] = partition
        supported.sort(key=lambda group: min(metadata[i]["start"] for i in group))
        if self.speaker_count is not None and len(supported) != self.speaker_count:
            raise SpeakerCountError("Недостаточно независимой речи для указанного количества голосов; сохранены предварительные назначения")
        if len(supported) > 15:
            supported = sorted(supported, key=lambda group: -sum(masses[i] for i in group))[:15]
        votes = [[sum(metadata[i]["votes"].get(key, 0.) for i in group) for key in old_keys]
                 for group in supported]
        mapping = stable_mapping(votes, old_keys)
        # Never remove or repurpose an ID selected explicitly by the user.
        protected = manual - set(mapping.values())
        next_number = max([self.speaker_offset, *[int(k.split("-")[1]) for k in self.speakers
                            if k.startswith("speaker-")]]) + 1
        gallery = {}
        gallery_origins = {}
        redirects = {}
        for index, group in enumerate(supported):
            key = mapping.get(index)
            if key is None:
                key = f"speaker-{next_number}"
                next_number += 1
                self.speakers[key] = Speaker(key, f"Спикер {key.split('-')[1]}")
            # A bounded, diverse gallery, rather than all observations of a
            # dominant person drowning out a short-lived participant.
            trusted = [i for i in group if anchors[i] is not None]
            if trusted:
                for i in group:
                    if anchors[i] is not None:
                        continue
                    scores = sorted((max(float(representatives[i] @ ref) for ref in self.references[old])
                                     for old in old_keys), reverse=True)
                    runner = scores[1] if len(scores) > 1 else 0.
                    sufficient = metadata[i]["longest"] >= 1. or (
                        metadata[i]["seconds"] >= 2.4 and metadata[i]["turns"] >= 3)
                    if sufficient and scores[0] >= self.match and scores[0] - runner >= .15:
                        trusted.append(i)
            chosen = sorted(trusted or group, key=lambda i: -masses[i])
            refs = [representatives[chosen[0]].copy()]
            origins = [metadata[chosen[0]]["turn_id"]]
            for i in chosen[1:]:
                if len(refs) < 8 and max(float(representatives[i] @ ref) for ref in refs) < .9:
                    refs.append(representatives[i].copy())
                    origins.append(metadata[i]["turn_id"])
            gallery[key] = refs
            gallery_origins[key] = origins
            for i in group:
                if anchors[i] is not None and anchors[i] != key and anchors[i] not in manual:
                    redirects[anchors[i]] = key
        if not gallery:
            return
        assignments = {}
        keys = list(gallery)
        centers = np.stack([gallery_center(gallery[key]) for key in keys])
        for turn in ordered:
            if turn.id not in observations or turn.manual_speaker_id is not None:
                continue
            vector = observations[turn.id]
            short = evidence[turn.id].speech_seconds < 1.
            scores = np.array([max((float(vector @ ref) for ref, origin in zip(gallery[key], gallery_origins[key])
                                    if not short or origin != turn.id), default=-1.) for key in keys])
            best = int(np.argmax(scores))
            runner = max((s for i, s in enumerate(scores) if i != best), default=0.)
            short_match = .66 if self.speaker_count is not None else .56
            threshold = max(short_match, self.match) if short else self.match
            margin = .20 if short else .15
            center_scores = centers @ vector
            center_agrees = self.speaker_count is None or int(np.argmax(center_scores)) == best
            if scores[best] >= threshold and scores[best] - runner >= margin and center_agrees:
                assignments[turn.id] = keys[best]
            elif self.speaker_count is not None:
                assignments[turn.id] = "unknown"
        for turn in turns:
            if turn.id in assignments:
                turn.speaker_id = assignments[turn.id]
                turn.attribution = "uncertain" if turn.speaker_id == "unknown" else "automatic"
            elif (self.speaker_count is not None and turn.manual_speaker_id is None
                  and (not self.speaker_offset or turn.source == "system")
                  and self.resolve_speaker(turn.speaker_id) not in gallery):
                turn.speaker_id, turn.attribution = "unknown", "uncertain"
        # Replace merged galleries only after all classification has succeeded.
        self.references = {**{k: self.references[k] for k in protected if k in self.references}, **gallery}
        self.reference_origins = {**{k: self.reference_origins.get(k, [None] * len(self.references[k]))
                                    for k in protected if k in self.references}, **gallery_origins}
        self.centers = {}
        for key, refs in self.references.items():
            self.centers[key] = gallery_center(refs)
        self.counts = {key: self.counts.get(key, 1) for key in self.centers}
        self.aliases.update(redirects)

    def _refine_known(self, turns, evidence):
        """Revisit uncertain turns against a frozen, meeting-wide voice gallery.

        Only acoustic summaries of transcribed pieces survive until this pass.
        Weak assignments never train the gallery or create/merge speakers.
        Evidence keys also restrict the pass to the diarized online channel.
        """
        gallery = {key: list(refs) for key, refs in self.references.items()}
        origins = {key: list(self.reference_origins.get(key, [None] * len(refs)))
                   for key, refs in self.references.items()}
        sums, weights = {}, {}
        for turn in turns:
            item = evidence.get(turn.id)
            key = self.resolve_speaker(turn.speaker_id)
            if item is None or key not in gallery or turn.manual_speaker_id is not None:
                continue
            if item.speech_seconds < 1.0:
                continue
            reference = np.stack(self.references[key])
            scores = (item.vectors @ reference.T).max(axis=1)
            good = scores >= self.match
            if not good.any():
                continue
            vector = unit(np.average(item.vectors[good], axis=0, weights=item.weights[good]))
            weight = min(item.speech_seconds, 4.0)
            sums[key] = sums.get(key, np.zeros(192, np.float32)) + vector * weight
            weights[key] = weights.get(key, 0.0) + weight
            for candidate in item.vectors[good]:
                if max(float(candidate @ ref) for ref in gallery[key]) < .9:
                    gallery[key] = [gallery[key][0], *gallery[key][-14:], candidate]
                    origins[key] = [origins[key][0], *origins[key][-14:], turn.id]
        for key in gallery:
            center = (unit(sums[key] / weights[key]) if key in sums and np.linalg.norm(sums[key]) >= 1e-8
                      else self.centers[key])
            gallery[key] = np.stack([center, *gallery[key]])
            origins[key] = [None, *origins[key]]

        keys = list(gallery)
        shared_centers = np.stack([gallery_center(gallery[key]) for key in keys]) if (gallery and self.speaker_count is not None) else None

        def choose(items, turn_ids):
            if not gallery:
                return "unknown"
            vectors = np.concatenate([item.vectors for item in items])
            weight = np.concatenate([item.weights for item in items])
            owners = np.concatenate([np.full(len(item.vectors), turn_id) for item, turn_id in zip(items, turn_ids)])
            short = np.concatenate([np.full(len(item.vectors), item.speech_seconds < 1.) for item in items])
            scores = np.stack([np.where(short[:, None] & (owners[:, None] == np.asarray(origins[key], object)),
                                        -1., vectors @ gallery[key].T).max(axis=1) for key in keys], axis=1)
            totals = np.average(scores, axis=0, weights=weight)
            best = int(np.argmax(totals))
            runner = max((score for i, score in enumerate(totals) if i != best), default=0.0)
            seconds = sum(item.speech_seconds for item in items)
            # Short evidence needs a larger lead over the other speaker.
            # Only acoustically consistent replies can pool their evidence.
            short_match = .66 if self.speaker_count is not None else .56
            threshold = max(self.new + .12, self.match - .08) if seconds >= 1.0 else max(short_match, self.match)
            margin = .15 if seconds >= 1.0 else .20
            agreement = np.average(np.argmax(scores, axis=1) == best, weights=weight)
            center_agrees = True
            if self.speaker_count is not None and shared_centers is not None:
                center_scores = np.average(vectors @ shared_centers.T, axis=0, weights=weight)
                center_agrees = int(np.argmax(center_scores)) == best
            if totals[best] < threshold or totals[best] - runner < margin or agreement < .7 or not center_agrees:
                return "unknown"
            return keys[best]

        groups = []
        previous = None
        for turn in sorted(turns, key=lambda turn: (turn.start, turn.id)):
            provisional = turn.speaker_id.startswith("pending-")
            turn.speaker_id = self.resolve_speaker(turn.speaker_id)
            if provisional:
                turn.attribution = "uncertain" if turn.speaker_id == "unknown" else "automatic"
            item = evidence.get(turn.id)
            if turn.speaker_id != "unknown" or turn.manual_speaker_id is not None or item is None:
                previous = None
                continue
            if (previous is not None and turn.source == previous.source
                    and 0 <= turn.start - previous.end <= 2.0
                    and float(unit(np.average(item.vectors, axis=0, weights=item.weights)) @
                              unit(np.average(evidence[previous.id].vectors, axis=0,
                                              weights=evidence[previous.id].weights))) >= .65):
                groups[-1].append(turn)
            else:
                groups.append([turn])
            previous = turn
        changed = 0
        for group in groups:
            key = choose([evidence[turn.id] for turn in group], [turn.id for turn in group])
            for turn in group:
                # If the group remains uncertain, a strong individual phrase may
                # still be identifiable. No propagation through missing evidence.
                assigned = key if key != "unknown" else choose([evidence[turn.id]], [turn.id])
                if assigned != "unknown":
                    turn.speaker_id = assigned
                    turn.attribution = "automatic"
                    changed += 1
        return changed

    def split(self, segment):
        pcm = segment.pcm
        if len(pcm) < 9600:
            return [(segment, "unknown")]

        def extract(offset, window):
            speech = pcm[offset:offset + window]
            if segment.speech_mask is not None:
                speech = speech[segment.speech_mask[offset:offset + window]]
            if len(speech) < 9600 or (speech.max() < 1e-4 and speech.min() > -1e-4):
                return None, 0., 0.
            quality = .5 if np.mean(np.abs(speech) >= .99) > .01 else 1.
            return unit(self.embedder(speech)), len(speech) / 16000, quality

        def offsets_for(window, hop):
            offsets = list(range(0, max(1, len(pcm) - window + 1), hop))
            last = max(0, len(pcm) - window)
            if offsets[-1] != last:
                offsets.append(last)
            return offsets

        window = 32000
        offsets = offsets_for(window, 8000)
        records = [extract(i, window) for i in offsets]
        # Probe at higher resolution only when the coarse acoustics change.
        # Coarse windows spanning a transition never train the gallery.
        transition = any(a[0] is not None and b[0] is not None and float(a[0] @ b[0]) < .7
                         for a, b in zip(records, records[1:]))
        if transition:
            if segment.speech_mask is not None:
                # Silero's half-second hangover can bridge two short replies.
                # Cut an internal pause before extracting mixed embeddings.
                edges = np.flatnonzero(np.diff(np.r_[False, ~segment.speech_mask, False]))
                cuts = [(int(a) + int(b)) // 2 for a, b in zip(edges[::2], edges[1::2])
                        if b - a >= 3840 and a > 0 and b < len(pcm)]
                if cuts:
                    pieces = []
                    for a, b in zip([0, *cuts], [*cuts, len(pcm)]):
                        pieces.extend(self.split(SpeechSegment(segment.source,
                            segment.start + a / 16000, segment.start + b / 16000,
                            pcm[a:b], segment.speech_mask[a:b])))
                    return pieces
            window = 16000
            offsets = offsets_for(window, 4000)
            records = [extract(i, window) for i in offsets]
        labels = []
        for i, (vector, seconds, quality) in enumerate(records):
            if vector is None or seconds < 1.0:
                labels.append("unknown")
            else:
                labels.append(self.identify(vector, start=segment.start + offsets[i] / 16000,
                    end=segment.start + min(len(pcm), offsets[i] + window) / 16000,
                    seconds=seconds, quality=quality))
        labels = [self.aliases.get(key, key) for key in labels]
        original = labels.copy()
        for i in range(1, len(labels) - 1):
            # Don't erase a short, acoustically supported speaker switch.
            if (original[i - 1] == original[i + 1] and original[i] == "unknown"
                    and records[i][2] < .65):
                labels[i] = original[i - 1]
        runs = []
        left, first, current = 0, 0, labels[0]
        for i in range(1, len(labels)):
            acoustic_change = (labels[i] == current == "unknown" and records[i][0] is not None
                               and records[i - 1][0] is not None
                               and float(records[i][0] @ records[i - 1][0]) < .45
                               and offsets[i] - left >= 9600 and len(pcm) - offsets[i] >= 9600)
            if labels[i] != current or acoustic_change:
                boundary = min(len(pcm), (offsets[i - 1] + offsets[i] + window) // 2)
                runs.append((left, boundary, current, first, i))
                left, first, current = boundary, i, labels[i]
        runs.append((left, len(pcm), current, first, len(labels)))
        pieces = []
        for a, b, speaker, first, last in runs:
            if b <= a:
                continue
            mask = segment.speech_mask[a:b] if segment.speech_mask is not None else None
            indices = [i for i in range(first, last) if records[i][0] is not None]
            if len(indices) > 8:
                indices = [indices[i] for i in np.linspace(0, len(indices) - 1, 8, dtype=int)]
            summary = None
            if indices:
                intervals = np.asarray([[max(a, offsets[i]), min(b, offsets[i] + window)]
                                        for i in indices], np.float64) / 16000 + segment.start
                quality = np.asarray([records[i][2] * (min(b, offsets[i] + window) - max(a, offsets[i])) /
                                      min(window, len(pcm) - offsets[i]) for i in indices], np.float32)
                summary = SpeakerEvidence(np.stack([records[i][0] for i in indices]),
                    np.asarray([records[i][1] for i in indices], np.float32),
                    (np.count_nonzero(mask) if mask is not None else b - a) / 16000,
                    intervals, quality)
            if transition and len(runs) > 1:
                # Once a boundary is known, use the actual piece, not the
                # overlapping discovery windows which contain its neighbours.
                speech = pcm[a:b] if mask is None else pcm[a:b][mask]
                if len(speech) >= 9600 and (speech.max() >= 1e-4 or speech.min() <= -1e-4):
                    quality = .5 if np.mean(np.abs(speech) >= .99) > .01 else 1.
                    summary = SpeakerEvidence(unit(self.embedder(speech))[None],
                        np.asarray([len(speech) / 16000], np.float32), len(speech) / 16000,
                        np.asarray([[segment.start + a / 16000, segment.start + b / 16000]]),
                        np.asarray([quality], np.float32))
            pieces.append((SpeechSegment(segment.source, segment.start + a / 16000,
                segment.start + b / 16000, pcm[a:b], mask, summary), speaker))
        return pieces
