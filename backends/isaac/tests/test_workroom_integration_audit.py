"""CPU integration audit summarizes the catalog and keeps ordinary episode checks."""
from scenes.workrooms import catalog
from scripts.maintenance import audit_workroom_integration as audit


def test_catalog_summary_accepts_discoverable_engineering_tasks(monkeypatch):
    def summarized(definition):
        return {**definition, 'passed': True, 'checks': {}, 'errors': [], 'route': None}

    monkeypatch.setattr(audit, 'audit_task', summarized)
    result = audit.audit(expected_count=len(catalog.list_tasks(include_planned=True)))
    assert result['passed'], result['errors']
    assert result['family_counts']['laundry_room'] == 5
    assert result['checks']['agent_catalog_task_inventory']
    assert 'laundry_no_tasks' not in result['checks']


def test_ordinary_task_retains_robot_policy_and_episode_checks():
    definition = catalog.list_tasks('kitchen')[0]
    result = audit.audit_task(definition)
    assert result['passed'], result['errors']
    assert result['route'] == 'POST /episodes'
    assert result['robot']['joint_count'] == 29
    assert result['checks']['one_task_bound_policy']
    assert result['checks']['reset_joint_inventory']
    assert result['checks']['worker_public_contract']
    assert 'ordinary_episode_capability_rejected' not in result['checks']
