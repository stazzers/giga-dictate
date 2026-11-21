"""Bounded, deterministic session-only average-link cosine clustering.

No waveform, transcript, model downloads or optional numerical dependencies.
"""
import numpy as np

MAX_REPRESENTATIVES = 1024


def average_linkage(vectors, weights, *, threshold=.65, count=None, anchors=None):
    """Return representative groups, optionally cutting the tree at exactly K.

    Anchors from the same confirmed gallery start together in automatic mode.
    Observations never inherit this constraint from their provisional label.
    """
    size = len(vectors)
    if not size:
        return []
    if size > MAX_REPRESENTATIVES:
        raise ValueError("Слишком много акустических представителей")
    scores = np.clip(vectors @ vectors.T, -1, 1)
    mass = np.maximum(np.asarray(weights, np.float64), .01).copy()
    groups = {i: [i] for i in range(size)}
    np.fill_diagonal(scores, -np.inf)
    active_mask = np.ones(size, dtype=bool)

    def merge(a, b):
        total = mass[a] + mass[b]
        active_mask[a] = False
        active_mask[b] = False
        scores[a, active_mask] = (scores[a, active_mask] * mass[a] + scores[b, active_mask] * mass[b]) / total
        scores[active_mask, a] = scores[a, active_mask]
        mass[a] = total
        groups[a].extend(groups.pop(b))
        scores[b, :] = scores[:, b] = -np.inf
        scores[a, a] = -np.inf
        active_mask[a] = True

    if anchors is not None and count is None:
        first = {}
        for i, key in enumerate(anchors):
            if key is not None:
                if key in first:
                    merge(first[key], i)
                else:
                    first[key] = i
    while len(groups) > (count if count is not None else 1):
        a, b = np.unravel_index(int(np.argmax(scores)), scores.shape)
        if not np.isfinite(scores[a, b]) or (count is None and scores[a, b] < threshold):
            break
        merge(min(a, b), max(a, b))
    return list(groups.values())


def stable_mapping(votes, keys):
    """Maximum-weight one-to-one cluster/old-ID matching (at most 15 IDs).

    Hungarian assignment with dummy columns. Zero-overlap matches are omitted.
    This keeps final IDs stable without letting two clusters share an old ID.
    """
    if not len(votes) or not keys:
        return {}
    rows, old = len(votes), len(keys)
    costs = np.zeros((rows, old + rows), np.float64)
    costs[:, :old] = -np.asarray(votes)
    columns = costs.shape[1]
    u, v = np.zeros(rows + 1), np.zeros(columns + 1)
    p, way = np.zeros(columns + 1, int), np.zeros(columns + 1, int)
    for i in range(1, rows + 1):
        p[0], j0 = i, 0
        minimum, used = np.full(columns + 1, np.inf), np.zeros(columns + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            for j in range(1, columns + 1):
                if not used[j]:
                    cur = costs[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minimum[j]:
                        minimum[j], way[j] = cur, j0
            available = np.where(~used[1:])[0] + 1
            j1 = available[np.argmin(minimum[available])]
            delta = minimum[j1]
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]
    return {int(p[j] - 1): keys[j - 1] for j in range(1, old + 1)
            if p[j] and votes[p[j] - 1][j - 1] > 0}


def supported_split(vectors, weights, eligible):
    """Split a large linkage group only into two independently supported cores.

    A fixed-K tree cut can spend one of its clusters on a single outlier. Drop
    that outlier and look for a supported split of a larger group instead of
    forcing the outlier to be a person. Deterministic spherical refinement uses
    a small set of diverse, locally supported seeds.
    """
    size = len(vectors)
    if size < 2:
        return None
    weights = np.asarray(weights, np.float64)
    similarities = vectors @ vectors.T
    density = (similarities >= .45) @ weights
    candidates = []
    for i in np.argsort(-density, kind="stable"):
        if not candidates or max(similarities[i, j] for j in candidates) < .92:
            candidates.append(int(i))
        if len(candidates) == 8:
            break
    best, best_score = None, -np.inf
    for a in candidates:
        for b in candidates:
            if b <= a:
                continue
            centers = vectors[[a, b]].copy()
            labels = None
            scores = None
            for _ in range(12):
                scores = vectors @ centers.T
                updated = np.argmax(scores, axis=1)
                if labels is not None and np.array_equal(updated, labels):
                    break
                labels = updated
                groups = [np.flatnonzero(labels == i) for i in range(2)]
                if any(not len(group) for group in groups):
                    break
                for i, group in enumerate(groups):
                    mean = np.average(vectors[group], axis=0, weights=weights[group])
                    centers[i] = mean / max(float(np.linalg.norm(mean)), 1e-8)
                scores = None
            groups = [np.flatnonzero(labels == i).tolist() for i in range(2)]
            if all(group and eligible(group) for group in groups):
                if scores is None:
                    scores = vectors @ centers.T
                score = float(np.sum(weights * scores.max(axis=1)))
                if score > best_score:
                    best, best_score = groups, score
    return best
