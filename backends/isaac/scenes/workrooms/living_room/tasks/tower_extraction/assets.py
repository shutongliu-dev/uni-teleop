"""Task4 independent blocks; opt-in tall tower with physical collection trays."""
from scenes.workrooms.living_room.tower_assets import author_tower_task


def author_task(stage, root_path='/World/LivingTask', *, presentation=False, profile=None):
    return author_tower_task(stage, 'tower_extraction', root_path, presentation=presentation, profile=profile)
