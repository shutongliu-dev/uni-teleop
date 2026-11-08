"""Current budget-repaired inputs; development outcomes stay outside admission."""
from scenes.workrooms.laboratory.catalog_definition import task_definition

TASK = task_definition(
    'pipette_transfer',
    manifest_sha256='fbdb937c8a1b6a7771e266644e571eac2492c9c8bd3135ffbdc0b8c5dd614091',
    asset_version='v4',
    release_version='budget_v1',
)
