# Environment module — SO100PickAndPlaceEnvironment can be imported before SimulationApp
# since it uses deferred imports internally (same pattern as Arena's examples).
from isaaclab_arena_vla.environments.so100_pick_and_place import SO100PickAndPlaceEnvironment

__all__ = ["SO100PickAndPlaceEnvironment"]
