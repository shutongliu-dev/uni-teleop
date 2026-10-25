"""Photograph a frozen kitchen or static robot composition using native Isaac RTX.

Select one healthy physical device with CUDA_VISIBLE_DEVICES=GPU-....
All scene, source, camera and actual sample receipts use the common renderer.
"""
from scripts.media.render_card_room import main


if __name__=='__main__':
    raise SystemExit(main(expected_family='kitchen'))
