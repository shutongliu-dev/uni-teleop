"""Task-owned identity guard around the shared native sequence referee."""
from scenes.workrooms.study.evaluation import SequenceEvaluator as _SequenceEvaluator

TASK_ID = 'book_stacking'


class BookStackingEvaluator(_SequenceEvaluator):
    def __init__(self, task_spec, *, trusted_normalized_input=False):
        super().__init__(task_spec, trusted_normalized_input=trusted_normalized_input)
        if self.ident != TASK_ID:
            raise ValueError('A different study task configuration was supplied')


SequenceEvaluator = BookStackingEvaluator
