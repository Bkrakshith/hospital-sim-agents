"""Bodies: who moves the people and the robot. The backend seam of the sim service.

The sim service keeps one StubWorld for all the logic (objects, reach, audio, robot tasks, routing for
placement checks, observations). After it advances the sim clock it calls ``bodies.step(world, t)``,
and that call is the only place bodies move:

- ``StubBodies`` (``--sim stub``, every automated test): the StubWorld's own kinematics, as before.
- ``IsaacBodies`` (``wardzero_agents.isaac_host.bodies``): Isaac Sim owns the bodies and their poses
  are read back into the StubWorld every frame.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stub_world import StubWorld


class Bodies:
    """Moves (or reads back) the bodies to sim time t. Subclasses run the world logic too."""

    name = "bodies"

    def step(self, world: "StubWorld", t: float) -> None:
        raise NotImplementedError

    def on_intent(self, world: "StubWorld", intent) -> None:
        """An intent the world has accepted (its plan is built); a body may start acting on it now."""

    def close(self) -> None:
        pass


class StubBodies(Bodies):
    """The stand-in kinematics: StubWorld walks the people and drives the robot itself."""

    name = "stub"

    def step(self, world: "StubWorld", t: float) -> None:
        world.step(t)
