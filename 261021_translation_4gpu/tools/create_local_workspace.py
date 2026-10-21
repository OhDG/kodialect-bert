#!/usr/bin/env python3
"""Create a fresh local code workspace from an NFS/shared experiment checkout.

Only the explicit code whitelist is copied. Prepared inputs, environments,
caches, results, checkpoints, and archives must never follow this operation.
Requires Python 3.10 or newer; no third-party packages are needed.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


ROOT_FILES = (
    "run_parallel.py", "setup_environment.sh", "requirements.txt", "README.md",
    "validation_report.json", ".gitignore", ".gitattributes", "NFS_GUIDE.md",
)
REQUIRED_FILES = ("run_parallel.py", "setup_environment.sh", "requirements.txt", "README.md")
CODE_PATTERNS = {"engine": ("*.py", "*.json"), "tools": ("*.py",),
                 "tests": ("*.py",), "configs": ("*.json",)}


class WorkspaceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise WorkspaceError(message)


def is_link(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def code_files(source):
    require(source.is_dir() and not is_link(source), f"Invalid/linked source directory: {source}")
    files = {}
    for name in ROOT_FILES:
        path = source / name
        if os.path.lexists(path):
            require(path.is_file() and not is_link(path), f"Invalid/linked code file: {path}")
            files[name] = path
    for name in REQUIRED_FILES:
        require(name in files, f"Missing required code file: {source / name}")
    for folder, patterns in CODE_PATTERNS.items():
        directory = source / folder
        if not os.path.lexists(directory):
            continue
        require(directory.is_dir() and not is_link(directory), f"Invalid/linked code directory: {directory}")
        for pattern in patterns:
            for path in directory.glob(pattern):
                require(path.is_file() and not is_link(path), f"Invalid/linked code file: {path}")
                files[path.relative_to(source).as_posix()] = path
    return dict(sorted(files.items()))


def filesystem_type(path):
    """Identify Linux mount type without shell commands or extra dependencies."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    path, matches = path.resolve(), []
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        if len(fields) < 5 or not after.split():
            continue
        mount = fields[4]
        for escaped, literal in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
            mount = mount.replace(escaped, literal)
        mount = Path(mount)
        if path.is_relative_to(mount):
            matches.append((len(mount.parts), after.split()[0]))
    return max(matches)[1] if matches else None


def publish_directory(staging, destination):
    """Publish the complete directory atomically, refusing a concurrent collision."""
    if os.name == "nt":
        os.rename(staging, destination)  # Windows rename refuses existing directories.
        return
    require(sys.platform.startswith("linux"), "Atomic no-overwrite publishing requires Linux or Windows")
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    require(rename is not None, "Linux renameat2 is unavailable; destination was not published")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    # RENAME_NOREPLACE (1): never replace even an empty destination directory.
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def create_workspace(source, destination):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    require(not os.path.lexists(destination), f"Refusing to overwrite existing destination: {destination}")
    require(not destination.resolve().is_relative_to(source.resolve()),
            "Destination must be outside the source directory")
    files = code_files(source)
    mount_type = filesystem_type(destination)
    require(mount_type not in ("nfs", "nfs4", "cifs", "smb3"),
            f"Destination uses shared filesystem {mount_type}; choose local server storage")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".workspace-import-", dir=destination.parent))
    try:
        copied = {}
        for name, original in files.items():
            require(original.is_file() and not is_link(original), f"Source code changed into a link: {original}")
            output = staging / name
            output.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            with original.open("rb") as stream, output.open("xb") as target:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    target.write(block)
                    digest.update(block)
                    size += len(block)
            shutil.copystat(original, output, follow_symlinks=False)
            copied[name] = {"sha256": digest.hexdigest(), "size_bytes": size}
        origin = {"format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                  "source_directory": str(source), "destination_directory": str(destination),
                  "destination_filesystem": mount_type, "files": copied,
                  "scope": "code only; data/models/provenance/environments/caches/runs/archives excluded"}
        (staging / "workspace_origin.json").write_text(
            json.dumps(origin, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        publish_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"completed": True, "destination": str(destination), "code_files_copied": len(files),
            "destination_filesystem": mount_type, "origin_record": str(destination / "workspace_origin.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True,
                        help="New local directory; must not already exist")
    args = parser.parse_args(argv)
    try:
        result = create_workspace(Path(__file__).resolve().parents[1], args.destination)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (WorkspaceError, OSError) as exc:
        print(f"WORKSPACE ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
