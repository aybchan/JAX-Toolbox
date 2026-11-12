"""Bound a stopped Bazel disk cache before uploading it from a batch-mode CI job."""

import argparse
import os
from pathlib import Path
import stat


def trim_cache(root: Path, max_bytes: int) -> int:
    if max_bytes < 0:
        raise ValueError("max_bytes must be nonnegative")
    if root.is_symlink() or root.is_junction():
        raise ValueError(f"Cache root must not be a symlink or junction: {root}")
    if not root.exists():
        return 0
    entries = []
    def fail_walk(error):
        raise error

    for directory, directories, files in os.walk(root, followlinks=False, onerror=fail_walk):
        for name in directories + files:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or path.is_junction():
                raise ValueError(f"Unexpected symlink or junction in disk cache: {path}")
            if stat.S_ISREG(info.st_mode):
                relative = path.relative_to(root).as_posix()
                entries.append((info.st_mtime_ns, relative, info.st_size, path))
    # Bazel uses mtime for last access, including cache hits. Match its GC's
    # oldest-first ordering and AC-before-CAS tie break. Drop incomplete writes.
    total = sum(size for _, _, size, _ in entries)
    before = total
    entries.sort(key=lambda entry: (not entry[1].startswith("tmp/"), entry[0], entry[1]))
    for _, relative, size, path in entries:
        if total <= max_bytes and not relative.startswith("tmp/"):
            break
        path.unlink()
        total -= size
    print(f"Bazel disk cache: {before:,} -> {total:,} bytes (limit {max_bytes:,})")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--max-bytes", type=int, default=4 * 1024**3)
    args = parser.parse_args()
    trim_cache(args.cache_dir, args.max_bytes)
