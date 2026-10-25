"""Scene names shared by preview and PICO collection entry points."""

from functools import partial

from .basket_loading_env import MujocoG1BasketLoadingEnv
from .bucket_carry_env import MujocoG1BucketCarryEnv
from .chair_lean_env import MujocoG1ChairLeanEnv
from .door_elbow_env import MujocoG1DoorElbowEnv
from .empty_env import MujocoG1EmptyEnv
from .plush_carry_env import MujocoG1PlushCarryEnv
from .sweep_env import MujocoG1SweepEnv
from .univlat_env import MujocoG1UniVLaTEnv

UNIVLAT_SCENES = {
    "univlat_sweep": "table_sweeping",
    "univlat_basket": "basket_loading",
    "back_tap": "back_tap_walking",
    "human_hug": "human_robot_hugging",
    "composed_cleanup": "composed_cleanup",
}

G1_ENVIRONMENTS = {
    "empty": MujocoG1EmptyEnv,
    "sweep": MujocoG1SweepEnv,
    "chair_lean": MujocoG1ChairLeanEnv,
    "door_elbow": MujocoG1DoorElbowEnv,
    "basket_loading": MujocoG1BasketLoadingEnv,
    "bucket_carry": MujocoG1BucketCarryEnv,
    "plush_carry": MujocoG1PlushCarryEnv,
    **{
        name: partial(MujocoG1UniVLaTEnv, task) for name, task in UNIVLAT_SCENES.items()
    },
}
