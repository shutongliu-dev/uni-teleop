"""Native RTX views of the fixed furnished laundry room."""
from scripts.media.render_card_room import main

if __name__=='__main__':
    raise SystemExit(main(expected_family='laundry_room',camera_choices=('room','workcell','tabletop','appliances','storage')))
