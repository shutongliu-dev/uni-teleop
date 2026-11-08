"""Keep native bundle state private while the worker owns its single referee.

The generic bundle reader requires an observer. It does not own game authority:
the separately attached expansion observer consumes each existing native tick.
No object identity, hidden card identity or native pose is publicly projected.
"""


class NativeTaskObserver:
    def __init__(self, configuration):
        if configuration != {'projection': 'private_worker_only'}:
            raise ValueError('Card-game bundle state must remain private')

    def on_frame(self, frame):
        pass

    def on_observe(self):
        return {}
