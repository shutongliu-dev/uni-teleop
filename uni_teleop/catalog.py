"""Supported device combinations and reusable configuration fragments."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def recipes():
    return json.loads((ROOT / "configs/recipes.json").read_text())


def components():
    return json.loads((ROOT / "configs/components.json").read_text())


def select(*, backend=None, arm=None, hand=None, robot=None):
    filters = {"backend": backend, "arm": arm, "hand": hand, "robot": robot}
    return {name: recipe for name, recipe in recipes().items()
            if all(value is None or recipe[key] == value for key, value in filters.items())}


def load_profile(name):
    recipe = recipes().get(name)
    if recipe is None:
        raise ValueError(f"Unknown recipe: {name}; use 'list'")
    pieces = components()
    profile = {**recipe, "env": {}, "checks": [], "settings": [], "command": []}
    arguments = []
    for component in recipe["components"]:
        piece = pieces[component]
        arguments.extend(piece.get("args", []))
        profile["env"].update(piece.get("env", {}))
        profile["checks"].extend(piece.get("checks", []))
        profile["settings"].extend(piece["settings"])
    engine = recipe['backend'] if recipe['backend'] in ('isaac', 'mujoco') else 'mujoco'
    profile['cwd'] = '${ROOT}/backends/' + engine
    if recipe.get('runtime') == 'sonic':
        interpreter = '${ISAAC_PYTHON_LAUNCHER}' if recipe['backend'] == 'isaac' else '${SONIC_PYTHON}'
        profile['command'] = [interpreter, '-m', 'uni_teleop.runtime.sonic', '--backend', recipe['backend']]
    elif recipe["backend"] == "isaac":
        profile["command"] = ["${DEXBENCH_PYTHON}", "-B", "-m", "uni_teleop.runtime.isaac",
                              "--source", recipe["entry"], "--"]
        arguments[:0] = ["--workbench-only", "--physics-profile", "cpu_rigid_v1",
                         "--no-task-scoring", "--practice-contact-warnings", "--profile"]
        if name == "isaac-pico-manus":
            profile["env"]["PICO_TELEOP_ENTRYPOINT"] = "scripts.teleop.master_slave_isaac"
    elif recipe.get('runtime') == 'franka':
        profile['command'] = ['${SONIC_PYTHON}', '-B', '-m', 'uni_teleop.runtime.mujoco',
                              '--source', recipe['entry'], '--']
    else:
        profile["command"] = ["${SONIC_PYTHON}", "${ROOT}/backends/mujoco/scripts/" + recipe["entry"]]
    profile["command"].extend([*arguments, *recipe["args"]])
    return profile


def configure(name, path):
    profile = load_profile(name)
    example = json.loads((ROOT / "docs/host.example.json").read_text())
    document = {"recipe": name, "settings": {k: example[k] for k in profile["settings"]},
                "arguments": []}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    path.chmod(0o600)
    return document
