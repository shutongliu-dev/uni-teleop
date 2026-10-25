"""Native RTX photographs of the fixed study or its static robot preview."""
from scripts.media.render_card_room import main


if __name__ == '__main__':
    raise SystemExit(main(expected_family='study', camera_choices=('room', 'workcell', 'tabletop', 'shelf')))
