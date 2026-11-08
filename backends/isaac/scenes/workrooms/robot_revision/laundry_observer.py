"""Explicit immutable observer compatibility for pure native hand frames.

Publication must bind this helper and the corrected observer to its complete
frozen source snapshot. Historical publications never opt in implicitly.
"""
import hashlib
from pathlib import Path

VERSION = 'laundry_native_wrist_frames.v1'
OBSERVER_PATH = 'evaluator/native_observer.py'
SOURCE_PATH = 'scenes/workrooms/laundry_room/native_observer.py'
ORIGINAL_SHA256 = '4ed4bfaa8e4f7ef932a780d64901da4c31cbaabf114a7e3a9d058dea42e76db6'
PATCHED_SHA256 = '53aa5f3e3b377a5b53be5b6b10f2b4594a280fc70662b8b0a62082440e9e3383'


def patch_observer(original_bytes: bytes) -> bytes:
    """Return one exact self-contained observer, rejecting any unknown input."""
    if type(original_bytes) is not bytes or hashlib.sha256(original_bytes).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('Unrecognized frozen laundry observer; compatibility must not rewrite it')
    source = Path(__file__).resolve().parents[1]/'laundry_room'/'native_observer.py'
    result = source.read_bytes()
    if hashlib.sha256(result).hexdigest() != PATCHED_SHA256:
        raise ValueError('Laundry wrist-frame observer differs from the approved compatibility version')
    return result
