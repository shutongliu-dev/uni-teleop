"""Shared finite episode allocation ceilings, independent of task rules.

These are admission limits, not default episode budgets or recommendations.
An explicitly disabled time horizon retains its existing ``None`` semantics.
"""

MAX_EPISODE_SIM_TIME_S = 150.
MAX_EPISODE_WALL_TIME_S = 2400.
MAX_EPISODE_ACTIONS = 7000
