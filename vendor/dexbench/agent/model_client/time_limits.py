"""Optional duration limits: JSON null means no local time deadline."""
from __future__ import annotations

import argparse
import math


def optional_positive(value, label, *, maximum=None):
    if value is None:
        return None
    if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            or (maximum is not None and value > maximum)):
        raise ValueError(f'{label} must be null or finite and positive'
                         + (f' and at most {maximum}' if maximum is not None else ''))
    return float(value)


def optional_seconds(value):
    """CLI spelling; configurations retain actual JSON null, never infinity."""
    if value.lower() in ('none', 'null', 'disabled'):
        return None
    try:
        return optional_positive(float(value), 'seconds')
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('use positive seconds or none/null/disabled') from None


def minimum_limit(*values):
    enabled = [value for value in values if value is not None]
    return min(enabled) if enabled else None
