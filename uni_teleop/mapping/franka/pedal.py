"""Foot-operated clutch for PICO controllers, independent of the simulator."""

def toggle_pedal(mapper, frame, *, now_ns, view_ready):
    if mapper.armed:
        mapper.set_active(False)
        return 'paused'
    if (not view_ready or frame is None or not 0 <= now_ns-frame.host_ns <= 150_000_000
            or not frame.tracking.get('focused', True)
            or any(side not in frame.tracking.get('controllers', {}) for side in mapper.bindings.values())):
        mapper.set_active(False)
        return 'resume_rejected_tracking'
    # set_active clears pose anchors but intentionally preserves view rotation.
    mapper.set_active(True)
    return 'resumed_reanchor'
