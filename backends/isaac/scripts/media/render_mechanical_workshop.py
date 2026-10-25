"""Native RTX photographs of a pinned mechanical environment or static preview.

Shares the existing static renderer; never starts physics or model evaluation.
"""
from scripts.media.render_card_room import main


if __name__ == '__main__':
    raise SystemExit(main(expected_family='mechanical_assembly'))
