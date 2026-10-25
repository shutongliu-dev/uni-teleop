"""Task-local entrypoint for the read-only frozen kitchen referee."""
from ...evaluation import NativeTaskObserver, SequenceEvaluator

__all__ = ["NativeTaskObserver", "SequenceEvaluator"]
