"""Cutlery task entry to the shared kitchen authoring contract."""
from scenes.workrooms.kitchen.assets import author_task as _author


def author_task(stage, root_path='/World/KitchenTask', *, presentation=False):
    return _author(stage, 'utensil_sorting', root_path, presentation=presentation)
