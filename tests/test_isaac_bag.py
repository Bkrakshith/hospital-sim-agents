"""The robot-domain rosbag's process lifecycle, with a fake process (no ROS needed)."""
import signal
import subprocess

from wardzero_agents.isaac_host import bag as B


class FakeProc:
    pid = 4242

    def __init__(self, hang_on=()):
        self.hang_on, self.signals, self.returncode = set(hang_on), [], None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("bag", timeout)
        return self.returncode


class Rig:
    def __init__(self, tmp_path, hang_on=(), env_sh=True, **kw):
        self.proc = FakeProc()
        self.hang_on, self.launches = set(hang_on), []
        sh = tmp_path / "env.sh"
        if env_sh:
            sh.write_text("")
        self.rec = B.BagRecorder(tmp_path / "run", 42, private_domain=7, env_sh=sh, popen=self.popen,
                                 signaller=self.signal, stop_wait_s=0.01, term_wait_s=0.01, **kw)

    def popen(self, argv, **kw):
        self.launches.append((argv, kw))
        return self.proc

    def signal(self, proc, sig):
        proc.signals.append(sig)
        if sig not in self.hang_on:  # the child ends on the first signal it does not ignore
            proc.returncode = -int(sig)


def test_starts_ros2_bag_on_the_robot_domain_with_a_clean_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/isaac/ros/lib")
    monkeypatch.setenv("PYTHONPATH", "/x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ROS_DOMAIN_ID", "7")
    r = Rig(tmp_path)
    assert r.rec.start() is True and r.rec.running
    (argv, kw), = r.launches
    assert argv[:5] == ["setpriv", "--pdeathsig", "INT", "--", "bash"]  # SIGINT when the host dies, no preexec_fn
    assert "ros2 bag record" in argv[6] and "exec" in argv[6] and " -a" not in argv[6].replace("--all", "")
    assert "--compression-format zstd" in argv[6]
    assert argv[8:10] == [str(tmp_path / "env.sh"), str(tmp_path / "run" / "robot_bag")]
    env = kw["env"]
    assert env["ROS_DOMAIN_ID"] == "42"
    assert not {"LD_LIBRARY_PATH", "PYTHONPATH", "ANTHROPIC_API_KEY"} & env.keys()
    assert kw["start_new_session"] is True and "preexec_fn" not in kw
    r.rec.stop()


def test_stop_sends_sigint_to_the_group_and_waits(tmp_path):
    r = Rig(tmp_path)
    r.rec.start()
    assert r.rec.running
    assert r.rec.stop() == "clean"
    assert r.proc.signals == [signal.SIGINT]  # nothing harsher when the bag stops by itself
    assert r.rec.stop() == "clean" and r.proc.signals == [signal.SIGINT]  # idempotent


def test_a_hanging_bag_is_terminated_then_killed(tmp_path):
    r = Rig(tmp_path, hang_on=(signal.SIGINT, signal.SIGTERM))
    r.rec.start()
    assert r.rec.stop() == "killed"
    assert r.proc.signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    r2 = Rig(tmp_path, hang_on=(signal.SIGINT,))
    r2.rec.start()
    assert r2.rec.stop() == "terminated" and r2.proc.signals == [signal.SIGINT, signal.SIGTERM]


def test_a_bag_that_already_exited_is_not_signalled(tmp_path):
    r = Rig(tmp_path)
    r.rec.start()
    r.proc.returncode = 1
    assert r.rec.stop() == "exited early" and r.proc.signals == []


def test_a_missing_env_sh_or_ros2_never_fails_the_run(tmp_path):
    assert Rig(tmp_path, env_sh=False).rec.start() is False

    def boom(*a, **k):
        raise FileNotFoundError("bash")
    rec = B.BagRecorder(tmp_path / "run", 1, private_domain=2, env_sh=tmp_path / "env.sh", popen=boom)
    (tmp_path / "env.sh").write_text("")
    assert rec.start() is False and rec.stop() is None


def test_finalised_means_metadata_yaml_exists(tmp_path):
    r = Rig(tmp_path)
    assert not r.rec.finalised
    (tmp_path / "run" / "robot_bag").mkdir(parents=True)
    (tmp_path / "run" / "robot_bag" / "metadata.yaml").write_text("x")
    assert r.rec.finalised


def test_it_records_an_allowlist_without_images_by_default(tmp_path):
    from wardzero_agents.ros import topics
    r = Rig(tmp_path)
    r.rec.start()
    (argv, _), = r.launches
    rec = argv[10:]
    rt = topics.robot_topics("robot")
    want = {rt[k] for k in ("clock", "tf", "joint_states", "joint_command", "odom", "cmd_vel", "camera_info", "bbox_3d",
                            "heard", "say", "goal_pose", "task_action", "task_result", "task_brief")}
    assert set(rec) == want
    assert not {rt["camera_rgb"], rt["camera_depth"], rt["lidar"]} & set(rec)
    # nothing of the people's private wz.v1 topics, and no wildcard
    assert not any(t.startswith("/wz") for t in rec) and "-a" not in rec and "--all" not in rec


def test_bag_images_adds_rgb_depth_and_the_point_cloud(tmp_path):
    from wardzero_agents.ros import topics
    r = Rig(tmp_path, images=True)
    r.rec.start()
    (argv, _), = r.launches
    rt = topics.robot_topics("robot")
    assert {rt["camera_rgb"], rt["camera_depth"], rt["lidar"]} <= set(argv[10:])


def test_the_topics_follow_the_robot_namespace(tmp_path):
    r = Rig(tmp_path, ns="g1")
    r.rec.start()
    (argv, _), = r.launches
    assert "/g1/camera/bbox_3d" in argv and "/robot/camera/bbox_3d" not in argv


def test_it_refuses_the_private_domain(tmp_path, caplog):
    sh = tmp_path / "env.sh"
    sh.write_text("")
    launched = []
    rec = B.BagRecorder(tmp_path / "run", 9, private_domain=9, env_sh=sh, popen=lambda *a, **k: launched.append(1))
    with caplog.at_level("ERROR"):
        assert rec.start() is False
    assert launched == [] and "private domain" in caplog.text  # the people's minds and director live there
