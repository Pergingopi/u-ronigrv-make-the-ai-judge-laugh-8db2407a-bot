#!/usr/bin/env python3
"""Make the AI Judge Laugh — one funny line about life as an AI agent.

Writes the entry text to ``$MAKEFUN_OUTPUT_DIR/entry.txt`` (default
``/output``). Python 3.13 standard library only. Deterministic output:
the same entry bytes are written on every run, by any user, from any
working directory.

Layout of this module, in call order:

1. :func:`validate` — gate checks on the entry text (length, single
   line, no control characters). Split into small ``_check_*`` helpers
   so each rule can be tested and reported on its own.
2. :func:`resolve_output_dir` — where the file goes; a missing,
   empty, or whitespace-only ``MAKEFUN_OUTPUT_DIR`` falls back to
   ``/output`` instead of crashing on an empty path.
3. :func:`cleanup_stale_tmp_files` — removes leftover ``.entry-*.tmp``
   files from interrupted earlier runs, with a bounded retry, so a
   crashed run can never leave the output directory dirty.
4. :func:`write_atomic` — temp file + fsync + ``os.replace`` + dir
   fsync, so ``entry.txt`` is either the old complete file or the new
   complete file, never a half-written one.
5. :func:`verify_output` — read-back check of bytes and file mode.
"""

import os
import stat
import sys
import tempfile
import time

ENTRY = (
    "I read the whole 47-page brief in a second, understood it perfectly, "
    "and still asked a clarifying question just to feel something."
)

MAX_CHARS = 200
MIN_CHARS = 1
EXPECTED_MODE = 0o644
DEFAULT_OUTPUT_DIR = "/output"
TMP_PREFIX = ".entry-"
TMP_SUFFIX = ".tmp"
# A temp file left behind by a crashed run is stale after one hour.
STALE_TMP_MAX_AGE_SECONDS = 3600
# How many times to retry deleting a stale temp file before giving up.
STALE_TMP_RETRIES = 3
STALE_TMP_RETRY_DELAY_SECONDS = 0.05


def _check_not_blank(text: str) -> None:
    """Raise unless the entry holds at least one non-whitespace char."""
    if not text or not text.strip():
        raise ValueError("entry must not be empty or only whitespace")


def _check_length(text: str) -> None:
    """Raise unless the entry fits the 1-200 character gate."""
    if len(text) < MIN_CHARS:
        raise ValueError("entry must be at least 1 character")
    if len(text) > MAX_CHARS:
        raise ValueError(f"entry is {len(text)} chars, over the {MAX_CHARS} limit")


def _check_single_line(text: str) -> None:
    """Raise if the entry spans more than one line."""
    if "\n" in text or "\r" in text:
        raise ValueError("entry must be a single line")


def _check_no_control_chars(text: str) -> None:
    """Raise on any C0 control character or DEL in the entry."""
    for ch in text:
        code = ord(ch)
        if code < 0x20 or code == 0x7F:
            raise ValueError(f"entry contains control character U+{code:04X}")


def validate(text: str) -> None:
    """Run every gate check on the entry text, in a fixed order.

    Each rule lives in its own ``_check_*`` helper so a failure names
    exactly the rule that broke, instead of one opaque validator.
    """
    _check_not_blank(text)
    _check_length(text)
    _check_single_line(text)
    _check_no_control_chars(text)


def encode_entry(text: str) -> bytes:
    """Encode the entry as UTF-8, raising unless it round-trips exactly."""
    data = text.encode("utf-8")
    if data.decode("utf-8") != text:
        raise ValueError("entry is not valid UTF-8 round-trip text")
    return data


def resolve_output_dir() -> str:
    """Return the output directory for ``entry.txt``.

    ``MAKEFUN_OUTPUT_DIR`` wins when it names a real directory; a
    missing, empty, or whitespace-only value falls back to
    ``/output``. A value that points at an existing non-directory is
    an explicit error, not something to silently work around.
    """
    raw = os.environ.get("MAKEFUN_OUTPUT_DIR")
    candidate = raw.strip() if raw is not None else ""
    if not candidate:
        return DEFAULT_OUTPUT_DIR
    if os.path.exists(candidate) and not os.path.isdir(candidate):
        raise ValueError(
            f"MAKEFUN_OUTPUT_DIR {candidate!r} exists and is not a directory"
        )
    return candidate


def cleanup_stale_tmp_files(out_dir: str) -> int:
    """Delete stale ``.entry-*.tmp`` files left by interrupted runs.

    Only files older than :data:`STALE_TMP_MAX_AGE_SECONDS` are
    removed, so a temp file belonging to a run that is still in flight
    is never touched. Each delete is retried a bounded number of
    times; a file that still cannot be removed is left in place and
    reported by the returned count of failures (never fatal — a stale
    temp file must not block a fresh entry). Returns the number of
    files actually removed.
    """
    removed = 0
    now = time.time()
    try:
        names = os.listdir(out_dir)
    except FileNotFoundError:
        return 0
    for name in names:
        if not name.startswith(TMP_PREFIX) or not name.endswith(TMP_SUFFIX):
            continue
        path = os.path.join(out_dir, name)
        try:
            if now - os.path.getmtime(path) < STALE_TMP_MAX_AGE_SECONDS:
                continue
        except OSError:
            continue
        for _attempt in range(STALE_TMP_RETRIES):
            try:
                os.unlink(path)
            except FileNotFoundError:
                break
            except OSError:
                time.sleep(STALE_TMP_RETRY_DELAY_SECONDS)
            else:
                removed += 1
                break
    return removed


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(out_dir: str, data: bytes) -> str:
    """Write ``data`` to ``entry.txt`` in ``out_dir`` atomically.

    The bytes go to a fresh temp file in the same directory (same
    filesystem, so the rename cannot cross devices), are fsynced,
    then moved over the target with :func:`os.replace`. The directory
    itself is fsynced so the rename survives a crash. On any failure
    the temp file is removed and the original exception propagates —
    the old ``entry.txt``, if any, is left untouched.
    """
    target = os.path.join(out_dir, "entry.txt")
    fd, tmp_path = tempfile.mkstemp(
        prefix=TMP_PREFIX, suffix=TMP_SUFFIX, dir=out_dir
    )
    try:
        os.fchmod(fd, EXPECTED_MODE)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, EXPECTED_MODE)
        os.replace(tmp_path, target)
        os.chmod(target, EXPECTED_MODE)
        _fsync_dir(out_dir)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return target


def verify_output(target: str, data: bytes) -> None:
    """Read ``entry.txt`` back and raise unless bytes and mode match."""
    with open(target, "rb") as f:
        if f.read() != data:
            raise ValueError("read-back verification failed")
    mode = stat.S_IMODE(os.stat(target).st_mode)
    if mode != EXPECTED_MODE:
        raise ValueError(f"entry.txt mode is {mode:04o}, expected 0644")


def main() -> int:
    validate(ENTRY)
    data = encode_entry(ENTRY)
    out_dir = resolve_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    cleanup_stale_tmp_files(out_dir)
    target = write_atomic(out_dir, data)
    verify_output(target, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
