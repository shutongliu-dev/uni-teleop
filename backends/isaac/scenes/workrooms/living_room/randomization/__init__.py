"""Versioned reset-only cases for the three ordinary living-room tasks.

This package does not control a robot or mutate a running physical scene.
"""

TASK_IDS = ('remote_operation', 'water_pouring', 'storage_box')
SEEDS = tuple(range(10))
CONFIGURATIONS = ('single_arm', 'dual_same_side', 'dual_opposed')
