"""Render a frozen laboratory with native Isaac RTX and source-bound receipts."""
from scripts.media.render_card_room import main


if __name__ == '__main__':
    raise SystemExit(main(expected_family='laboratory',
                          camera_choices=('room', 'workcell', 'tabletop', 'reverse', 'safety')))
