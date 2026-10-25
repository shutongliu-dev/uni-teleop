"""Task8 public rule entrypoint; frozen with the common NIST referee."""
from scenes.workrooms.mechanical_assembly import nist_rules

TASK_ID = 'nist_cable_routing'


def public_rules(*, seed=0):
    return nist_rules.public_rules(TASK_ID, seed=seed)


def thresholds():
    return nist_rules.thresholds(TASK_ID)
