"""Task-local entrypoint; physical composition remains shared by the study family."""
from ...composition import build_task_inputs as _build


def build_task_inputs(*, seed=0):
    return _build('reading_stand',seed=seed)
