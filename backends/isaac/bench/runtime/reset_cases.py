"""Dispatch frozen reset projections without exposing private case payloads."""

RANDOMIZED_RELEASE_SCHEMAS = (
    'living_room_randomized_release.v1',
    'workroom_randomized_release.v1',
    'laundry_randomized_release.v1',
)



def effective_release_schema(manifest):
    """Semantic dispatch only, after validate(); immutable identity stays outer."""
    if manifest.get('schema') == 'workroom_robot_revision_release.v1':
        return manifest['parent_schema']
    return manifest.get('schema')


def effective_release_version(manifest):
    if manifest.get('schema') == 'workroom_robot_revision_release.v1':
        return manifest['parent_schema_version']
    return manifest.get('schema_version')


def is_randomized_release(manifest):
    if manifest.get('schema') == 'workroom_robot_revision_release.v1':
        return (manifest.get('selection_kind') == 'finite_seed'
                and manifest.get('parent_schema') in RANDOMIZED_RELEASE_SCHEMAS)
    return manifest.get('schema') in RANDOMIZED_RELEASE_SCHEMAS

def project_reset_case(task, projection):
    case = task.get('reset_case')
    if isinstance(case, dict) and case.get('schema') == 'laundry_reset_case.v1':
        from scenes.workrooms.laundry_room.randomization.public import project_reset_case as project
    else:
        from scenes.workrooms.initial_randomization.public import project_reset_case as project
    return project(task, projection)
