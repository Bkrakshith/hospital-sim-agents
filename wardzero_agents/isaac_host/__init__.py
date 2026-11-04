"""The Isaac host: WardZero's world simulator on Isaac Sim 6.0 ("StubWorld logic + Isaac bodies").

Run with Isaac Sim's own Python through ``scripts/isaac_host.sh`` (``hhumans run --sim isaac`` does
that). It speaks the same wz.v1 contract on the private ROS domain as ``ros/stub_sim_node``, through
the same ``ros.sim_service.SimService``; only the Bodies differ.

- ``stage``: which hospital areas to keep (pure Python) and the culling wrapper layer (pxr).
- ``bodies``: ``IsaacBodies``, poses read back from Isaac into the StubWorld every frame.
- ``service``: ``HostLoop``, the frame loop (sim clock from the physics step, hold, pacing, stats).
- ``recording``, ``bag``, ``mp4fix``: ``--record`` (two MP4s, the robot-domain rosbag) and the Studio's ``--preview``.
- ``app``: the Kit app and the binding start-up order; the entry point.

Nothing here imports Isaac, pxr or rclpy at module level, so the logic is unit-tested in the venv.
The host never receives the Claude key.
"""
