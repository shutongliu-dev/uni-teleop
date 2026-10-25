"""Task9: independent NIST-derived task entrypoint."""
from scenes.workrooms.mechanical_assembly.nist_composition import build_task_inputs as _build
from scenes.workrooms.mechanical_assembly.nist_release import prepare_task_release as _prepare

TASK_ID = 'nist_wire_harness'


def build_task_inputs(**options):
    return _build(TASK_ID, **options)


def prepare_task_release(output, **options):
    return _prepare(TASK_ID, output, **options)
