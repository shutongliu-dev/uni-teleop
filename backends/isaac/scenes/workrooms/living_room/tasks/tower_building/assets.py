"""Task3 scene authoring; the three-layer presentation is not a scored reset."""
from scenes.workrooms.living_room.tower_assets import author_tower_task


def author_task(stage, root_path='/World/LivingTask', *, presentation=False, profile=None):
    return author_tower_task(stage, 'tower_building', root_path, presentation=presentation, profile=profile)
