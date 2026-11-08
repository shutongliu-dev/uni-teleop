"""All laundry discovery contracts must fit the shared Agent/Bench schema."""
import pytest

from bench.protocol.task_referee import PublicContract
from scenes.workrooms.laundry_room.rules import (
    ALL_TASK_IDS, ENGINEERING_CONTROL, ROBOT_CONTROL, public_rules)


@pytest.mark.parametrize('task_id', ALL_TASK_IDS)
@pytest.mark.parametrize('mode', [ENGINEERING_CONTROL, ROBOT_CONTROL])
def test_every_laundry_task_exposes_a_schema_valid_public_contract(task_id, mode):
    contract = public_rules(task_id, control_mode=mode)['referee']
    parsed = PublicContract.model_validate(contract)
    assert parsed.model_dump() == contract
