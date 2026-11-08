"""Inspect this scene candidate without creating an episode."""
from scenes.workrooms.card_room.expansion_tasks import candidate_specification as _specification, main as _main


def candidate_specification():
    return _specification('polyomino_territory')


def main():
    _main('polyomino_territory')


if __name__ == "__main__":
    main()
