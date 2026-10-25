"""Compatibility import for the scene-owned public manipulation contract.

Task-specific rule allowlists belong to the published scene; shared execution
only consumes the resulting public contract.
"""
from scenes.workrooms.living_room.public_task import project_manipulation_task, select

__all__ = ["project_manipulation_task", "select"]
