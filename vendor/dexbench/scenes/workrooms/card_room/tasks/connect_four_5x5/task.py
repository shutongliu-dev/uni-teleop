"""Level-specific entrypoint; environment and shared game composition stay separate."""
from scenes.workrooms.card_room.game_task import (
    default_assets, make_scene as _make_scene, main as _main, room_reference,
)

ASSETS = default_assets('connect_four_5x5')


def make_scene(asset_root=ASSETS, *, room=None):
    return _make_scene('connect_four_5x5', asset_root, room=room)


def main():
    _main('connect_four_5x5')


if __name__ == '__main__':
    main()
