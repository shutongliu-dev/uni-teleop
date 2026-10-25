"""Robot kinematics, geometric executors and reference-motion retargeting."""
__all__ = ['ClassicalFingertipPolicy', 'Fr3SharpaGeometry', 'GeometryIKError']


def __getattr__(name):
    # FK and motion preparation need only NumPy. Load the protocol-dependent
    # executor only when a caller requests one of its public exports.
    if name in __all__:
        from . import fingertip
        value = getattr(fingertip, name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
