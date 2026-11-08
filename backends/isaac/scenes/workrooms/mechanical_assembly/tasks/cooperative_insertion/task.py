"""Task10: one agent controls both arms for cooperative insertion."""
from scenes.workrooms.mechanical_assembly.cooperative_composition import build_task_inputs
from scenes.workrooms.mechanical_assembly.cooperative_release import prepare_task_release as _prepare
TASK_ID='cooperative_insertion'

def prepare_task_release(output,**options):return _prepare(TASK_ID,output,**options)
