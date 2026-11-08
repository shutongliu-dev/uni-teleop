"""Uniform frozen evaluator sources for explicit physical parameters."""
PUBLIC_MASS_TEXT_VERSION = 'solid_weighing_actual_mass_v1'


def checked_public_mass_text_version(manifest):
    """Absent means the exact historical source, never the latest generation."""
    if 'public_mass_text_version' not in manifest:
        return None
    version = manifest['public_mass_text_version']
    if (version != PUBLIC_MASS_TEXT_VERSION or manifest.get('family') != 'laboratory'
            or manifest.get('task_id') != 'solid_weighing'):
        raise ValueError('Unknown or inapplicable public mass text version')
    return version


def laboratory_rules(values, *, public_mass_text_version=None):
    """Bind public dry masses to the authored bodies; keep numerical task thresholds."""
    if public_mass_text_version is not None:
        checked_public_mass_text_version({'public_mass_text_version': public_mass_text_version,
            'family': 'laboratory', 'task_id': values['task']['id']})
    if values['task'].get('asset_metadata', {}).get('root_path') != '/World/LaboratoryTask':
        return
    sources = values['evaluator_sources']; rules = sources['rules.py']
    if 'def object_properties(task_id):' in rules:
        rules = rules.replace('def object_properties(task_id):','def object_properties(task_id, metadata=None):')
        rules = rules.replace('    basis = (',
            "    if metadata is not None:\n        objects = {row['id']: row for row in metadata['objects']}\n        if set(objects) != {row[0] for row in rows}:\n            raise ValueError('Physical nominal mass inventory differs')\n        rows = [(name, label, objects[name]['mass_kg']) for name, label, _ in rows]\n    basis = (")
        rules = rules.replace('def public_rules(task_id, seed=0):','def public_rules(task_id, seed=0, metadata=None):')
        rules = rules.replace("'object_properties': object_properties(task_id)","'object_properties': object_properties(task_id, metadata)")
        rules = rules.replace('rules = public_rules(task_id, seed)','rules = public_rules(task_id, seed, metadata)')
    else:
        rules = rules.replace("if metadata is not None and metadata.get('geometry_variant', {}).get('schema') == 'laboratory_seed_geometry.v1':", 'if metadata is not None:')
    task = values['task']; md = task['asset_metadata']
    marker = '# Explicit parameterized public mass/count projection.'
    if marker not in rules:
        rules += r'''
# Explicit parameterized public mass/count projection.
_parameterized_public_rules = public_rules

def public_rules(task_id, seed=0, metadata=None):
    result = _parameterized_public_rules(task_id, seed, metadata)
    if task_id == 'solid_weighing' and metadata is not None:
        p = metadata['task_parameters']; mass = p['pellet_mass_kg']
        low = math.ceil((p['target_mass_kg']-p['tolerance_mass_kg'])/mass)
        high = math.floor((p['target_mass_kg']+p['tolerance_mass_kg'])/mass)
        if not 0 <= low <= high <= len(p['pellet_ids']):
            raise ValueError('Sampled pellet mass cannot achieve unchanged weighing target')
        def replace(value):
            if isinstance(value, str): return value.replace('11–13颗', str(low)+'–'+str(high)+'颗')
            if isinstance(value, dict): return {key:replace(v) for key,v in value.items()}
            if isinstance(value, list): return [replace(v) for v in value]
            return value
        result = replace(result)
    return result
'''
    # Keep the entire old frozen source above byte-identical. Only an explicit
    # immutable asset-manifest version selects this additional public projection.
    if public_mass_text_version == PUBLIC_MASS_TEXT_VERSION and '# Actual-mass weighing text v1.' not in rules:
        rules += r'''
# Actual-mass weighing text v1.
_parameterized_mass_text_public_rules = public_rules

def public_rules(task_id, seed=0, metadata=None):
    result = _parameterized_mass_text_public_rules(task_id, seed, metadata)
    if task_id == 'solid_weighing' and metadata is not None:
        p = metadata['task_parameters']; mass = p['pellet_mass_kg']
        spill_count = math.floor(p['spill_mass_limit_kg']/mass)
        cup_mass = next(row['mass_kg'] for row in metadata['objects'] if row['id'] == 'weighing_cup')
        replacements = {
            '1克（2颗）': format(p['spill_mass_limit_kg']*1000, '.6g')+'克（'+str(spill_count)+'颗）',
            '扣除28克空舟': '扣除'+format(cup_mass*1000, '.6g')+'克空舟',
        }
        def replace(value):
            if isinstance(value, str):
                for before, after in replacements.items(): value = value.replace(before, after)
                return value
            if isinstance(value, dict): return {key:replace(v) for key,v in value.items()}
            if isinstance(value, list): return [replace(v) for v in value]
            return value
        result = replace(result)
    return result
'''
    sources['rules.py']=rules
    evaluator=sources['sequence_evaluator.py']
    evaluator=evaluator.replace("public_rules(task['id'], task['episode_seed'])","public_rules(task['id'], task['episode_seed'], c['metadata'])")
    evaluator=evaluator.replace("public_rules(self.c['task_id'], self.c['episode_seed'])","public_rules(self.c['task_id'], self.c['episode_seed'], self.c['metadata'])")
    sources['sequence_evaluator.py']=evaluator
    namespace={}; exec(compile(rules,'<frozen laboratory rules>','exec'),namespace)
    task['public_rules']=namespace['public_rules'](task['id'],task['episode_seed'],md)
    namespace['bind_task_rules'](task)
    if 'public_rules_sha256' in task.get('evaluation',{}):
        task['evaluation']['public_rules_sha256']=namespace['digest'](task['public_rules'])


def study_rules(values):
    from pathlib import Path
    task=values['task'];task_id=task['id']
    if task_id not in ('drawer_retrieval','reading_stand'):return
    filename='drawer_rules.py' if task_id=='drawer_retrieval' else 'reading_stand_rules.py'
    sources=values['evaluator_sources'];marker='# Explicit sampled study mechanism configurations.'
    if marker not in sources[filename]:
        sources[filename]+='''
# Explicit sampled study mechanism configurations.
_parameterized_original_validate = validate_configuration

def validate_configuration(value):
    if isinstance(value, dict) and 'parameterized_physics' in value:
        from .parameterized_study import validate_configuration as validate_parameterized
        return validate_parameterized(value, configuration())
    return _parameterized_original_validate(value)
'''
    directory=Path(__file__).parent
    # Published study evaluators freeze the sampler as it was before the layout
    # amplitude fallback; study_evaluator_spec.py keeps exactly those bytes.
    sources['parameterized_spec.py']=(directory/'study_evaluator_spec.py').read_text()
    sources['parameterized_study.py']=(directory/'study_configuration.py').read_text().replace('from .spec import','from .parameterized_spec import')
