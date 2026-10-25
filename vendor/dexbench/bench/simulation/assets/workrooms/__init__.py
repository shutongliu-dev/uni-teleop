"""OpenUSD room authoring recipes; import the builder only in a USD environment.

This package has no Isaac runtime dependency. Its renderer is a separate native
tool; existing published room USDs remain immutable shared assets.
"""

FAMILIES = ('mechanical_assembly', 'laboratory', 'kitchen', 'study', 'card_room', 'living_room')

SOURCE_RESOURCES = (
    'scenes/workrooms/current_release.json',
    'scenes/catalog.json',
    'bench/apps/workroom.headless.kit',
)


def shared_asset_links(project):
    """Explicitly preserve the existing optional shared Shadow asset reference."""
    name = 'bench/simulation/assets/shadow_hand'
    return (name,) if (project / name).is_symlink() else ()
