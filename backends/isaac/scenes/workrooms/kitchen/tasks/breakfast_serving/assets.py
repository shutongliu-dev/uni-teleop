"""Breakfast task entry to the shared kitchen authoring contract."""
from scenes.workrooms.kitchen.assets import author_task as _author


def author_task(stage, root_path='/World/KitchenTask', *, presentation=False):
    return _author(stage, 'breakfast_serving', root_path, presentation=presentation)
