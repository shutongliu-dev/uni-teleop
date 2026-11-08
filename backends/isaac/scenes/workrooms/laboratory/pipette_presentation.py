"""Ordinary Task5 camera framing; no diagnostic state or hidden water readout."""


def pipette_camera_overrides():
    """Show the source, barrel scale and real piston on the island.

    The existing overview retains both worktops and receiver cups; public wrist
    cameras can inspect the rear dispensing operation after the hand moves.
    """
    return {'eye': [.98, .10, 1.28], 'target': [.73, .53, 1.015],
            'focal_length_mm': 45., 'title_zh': '移液视角（供液杯、筒身刻度与活塞）'}


def receiver_camera_overrides():
    """Ordinary receiver view makes the real 8/10/12 mL levels inspectable."""
    return {'eye': [.72, 1.68, 1.16], 'target': [.72, 2.08, .934],
            'focal_length_mm': 35., 'horizontal_aperture_mm': 36.,
            'vertical_aperture_mm': 27., 'title_zh': '分装视角（接收杯 A/B 与液位刻度）'}
