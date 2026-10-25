"""Task-owned entrypoint for the shared frozen study rule contract."""
from scenes.workrooms.study.rules import (OBSERVATION_SCHEMA, RULES_VERSION, THRESHOLDS,
    build_configuration as _build_configuration, public_rules as _public_rules)

TASK_ID = 'book_shelving'


def build_configuration(objects, targets, supports, **kwargs):
    return _build_configuration(TASK_ID, objects, targets, supports, **kwargs)


def public_rules(configuration=None):
    if configuration is not None and configuration.get('task_id') != TASK_ID:
        raise ValueError('A different study task configuration was supplied')
    return _public_rules(TASK_ID, configuration)
