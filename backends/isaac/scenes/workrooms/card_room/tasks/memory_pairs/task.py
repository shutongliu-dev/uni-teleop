"""Inspect this scene candidate without creating an episode."""
from scenes.workrooms.card_room.expansion_tasks import candidate_specification as _specification, main as _main


def candidate_specification():
    return _specification('memory_pairs')


def main():
    _main('memory_pairs')


if __name__ == "__main__":
    main()
