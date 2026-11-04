from wardzero_agents.ros.fairness import audit, live_audit


def test_audit_flags_robot_reading_minds():
    v = audit({"reception_bot": ["/g1/audio/heard", "/humans/P1/intent"],
               "wardzero_director": ["/humans/P1/status"]})
    assert v == [{"node": "reception_bot", "topic": "/humans/P1/intent"}]


def test_audit_clean():
    assert audit({"reception_bot": ["/g1/audio/heard"]}) == []


def test_audit_every_forbidden_prefix():
    topics = ["/humans/roster", "/scenario/control", "/sim/audio/events", "/sim/world_event",
              "/agents/pending", "/g1/audio/heard", "/g1/speech/say", "/clock"]
    v = audit({"robot_stack": topics})
    assert [x["topic"] for x in v] == topics[:5]


def test_audit_allowed_prefix_uses_the_node_name_not_the_namespace():
    v = audit({"/sim/wardzero_agents": ["/humans/P1/status"],
               "/wardzero_ns/spy": ["/humans/P1/status"]})
    assert v == [{"node": "/wardzero_ns/spy", "topic": "/humans/P1/status"}]


class FakeNode:
    def __init__(self, graph):
        self.graph = graph  # {(name, ns): [topics]}

    def get_node_names_and_namespaces(self):
        return list(self.graph)

    def get_subscriber_names_and_types_by_node(self, name, ns):
        if (name, ns) not in self.graph:
            raise RuntimeError("node vanished")
        return [(t, ["std_msgs/msg/String"]) for t in self.graph[(name, ns)]]


def test_live_audit_reads_the_graph():
    node = FakeNode({("reception_bot", "/"): ["/g1/audio/heard", "/humans/P2/observation"],
                     ("wardzero_agents", "/"): ["/humans/P2/observation"],
                     ("spy", "/lab"): ["/scenario/timeline"]})
    assert live_audit(node) == [
        {"node": "reception_bot", "topic": "/humans/P2/observation"},
        {"node": "/lab/spy", "topic": "/scenario/timeline"},
    ]


def test_live_audit_skips_nodes_that_vanish():
    class Flaky(FakeNode):
        def get_node_names_and_namespaces(self):
            return [("gone", "/"), *self.graph]

    assert live_audit(Flaky({("reception_bot", "/"): ["/g1/audio/heard"]})) == []


def test_fairness_flags_new_forbidden_topics():
    from wardzero_agents.ros.topics import robot_topics
    live = list(robot_topics("robot").values())
    assert audit({"my_robot_stack": live}) == []  # the whole robot-side table is allowed
    v = audit({"my_robot_stack": live + ["/humans/P1/status", "/humans/roster"]})
    assert [x["topic"] for x in v] == ["/humans/P1/status", "/humans/roster"]
