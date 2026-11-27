#!/usr/bin/env python3
"""Verify arm64 build targets and dyld dependencies of every native bundle file."""

import argparse
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


MACH_O_MAGIC = {
    bytes.fromhex(value) for value in ("feedface", "cefaedfe", "feedfacf", "cffaedfe",
                                      "cafebabe", "bebafeca", "cafebabf", "bfbafeca")
}
LOAD_DYLIB = {"LC_LOAD_DYLIB", "LC_LOAD_WEAK_DYLIB", "LC_REEXPORT_DYLIB",
              "LC_LOAD_UPWARD_DYLIB", "LC_LAZY_LOAD_DYLIB"}


def native_info(path):
    output = subprocess.check_output(
        ["xcrun", "otool", "-arch", "arm64", "-l", str(path)],
        stderr=subprocess.STDOUT, text=True,
    )
    target, platform, dependencies, rpaths = None, None, [], []
    for block in re.split(r"(?m)^Load command \d+\n", output)[1:]:
        fields = dict((key, value.strip()) for key, value in re.findall(
            r"(?m)^\s*(cmd|platform|minos|version|name|path)\s+(.+)$", block))
        command = fields.get("cmd")
        if command == "LC_BUILD_VERSION":
            target, platform = fields["minos"], fields["platform"]
        elif command == "LC_VERSION_MIN_MACOSX":
            target, platform = fields["version"], "MACOS"
        elif command in LOAD_DYLIB:
            dependencies.append(fields["name"].split(" (offset ", 1)[0])
        elif command == "LC_RPATH":
            rpaths.append(fields["path"].split(" (offset ", 1)[0])
    if target is None or platform.upper() not in ("MACOS", "1"):
        raise ValueError("missing arm64 macOS deployment target")
    return target, dependencies, rpaths


def version(value):
    return tuple(map(int, (value + ".0.0").split(".")[:3]))


def expand(path, loader, executable):
    if path == "@loader_path":
        return loader.parent
    if path == "@executable_path":
        return executable.parent
    if path.startswith("@loader_path/"):
        return loader.parent / path.removeprefix("@loader_path/")
    if path.startswith("@executable_path/"):
        return executable.parent / path.removeprefix("@executable_path/")
    return Path(path)


def check_bundle(bundle, minimum, launcher, *, strip_external_rpaths=False):
    native = []
    for path in bundle.rglob("*"):
        if not path.is_symlink() and path.is_file():
            with path.open("rb") as handle:
                if handle.read(4) in MACH_O_MAGIC:
                    native.append(path)
    if launcher not in native:
        raise ValueError("missing native GigaDictate launcher")

    def inspect(path):
        try:
            target, dependencies, rpaths = native_info(path)
            if strip_external_rpaths:
                for item in tuple(rpaths):
                    directory = expand(item, path, launcher)
                    if not directory.is_absolute() or not directory.resolve().is_relative_to(bundle):
                        subprocess.check_output(
                            ["xcrun", "install_name_tool", "-delete_rpath", item, str(path)],
                            stderr=subprocess.STDOUT, text=True,
                        )
                        rpaths.remove(item)
            return path, (target, dependencies, rpaths), None
        except (subprocess.CalledProcessError, KeyError, ValueError) as error:
            return path, None, getattr(error, "output", None) or str(error)

    errors, metadata = [], {}
    with ThreadPoolExecutor(max_workers=4) as workers:
        for path, info, error in workers.map(inspect, sorted(native)):
            if error:
                errors.append(f"{path.relative_to(bundle)}: {error}")
            else:
                metadata[path] = info
    if launcher not in metadata:
        raise ValueError("\n".join(errors))
    executable_rpaths = [expand(item, launcher, launcher) for item in metadata[launcher][2]]

    for path, (target, dependencies, rpaths) in metadata.items():
        label = path.relative_to(bundle)
        if version(target) > version(minimum):
            errors.append(f"{label}: requires macOS {target}; expected {minimum} or earlier")
        paths = [expand(item, path, launcher) for item in rpaths]
        for directory in paths:
            if not directory.is_absolute() or not directory.resolve().is_relative_to(bundle):
                errors.append(f"{label}: external dyld runpath {directory}")
        for dependency in dependencies:
            if dependency.startswith(("/usr/lib/", "/System/Library/")):
                continue
            if dependency.startswith("@rpath/"):
                relative = dependency.removeprefix("@rpath/")
                candidates = [directory / relative for directory in (*paths, *executable_rpaths)]
            elif dependency.startswith(("@loader_path/", "@executable_path/")):
                candidates = [expand(dependency, path, launcher)]
            else:
                candidates = []
            if not any(candidate.resolve() in metadata for candidate in candidates):
                errors.append(f"{label}: unresolved or external dependency {dependency}")
    if errors:
        raise ValueError("\n".join(errors))
    return native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("minimum")
    parser.add_argument("--strip-external-rpaths", action="store_true",
                        help="Remove build-machine runpaths from a staged bundle before verification")
    parser.add_argument("--signing-list", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    try:
        native = check_bundle(bundle, args.minimum, bundle / "Contents/MacOS/GigaDictate",
                              strip_external_rpaths=args.strip_external_rpaths)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    if args.signing_list:
        args.signing_list.write_bytes(b"".join(os.fsencode(path) + b"\0" for path in native))
    print(f"Verified arm64 and macOS {args.minimum} compatibility metadata for {len(native)} Mach-O files.")


if __name__ == "__main__":
    main()
