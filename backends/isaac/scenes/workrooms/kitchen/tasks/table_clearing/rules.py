"""Task-local public entrypoint; shared kitchen contract is frozen with the release."""
from ...rules import public_rules as _public_rules, evaluation_configuration as _configuration


def public_rules(seed=0):
    return _public_rules('table_clearing', seed)


def evaluation_configuration(metadata, seed=0, physics_dt_s=.0005):
    return _configuration('table_clearing', metadata, seed, physics_dt_s)
