"""Flows as Mermaid diagrams and as the document docs/scenario-flows.md (made by scripts/flows_doc.py)."""
from __future__ import annotations

from .spec import Flow

DOC_SKIP = ("scrub_nurse",)  # presets the public document leaves out
KIND_WORD = {"wait": "Wait", "wait_until": "Wait until", "expect": "Expect", "fire": "Event"}


def _q(text: str) -> str:
    return '"' + str(text).replace('"', "'").replace("\n", " ") + '"'


class _Mermaid:
    def __init__(self):
        self.lines = ["flowchart TD"]
        self.n = 0

    def node(self, text: str, shape: str = "box") -> str:
        self.n += 1
        nid = f"n{self.n}"
        l, r = {"box": ("[", "]"), "event": ("[[", "]]"), "expect": ("([", "])"), "split": ("{{", "}}"),
                "decide": ("{", "}"), "end": ("((", "))")}[shape]
        self.lines.append(f"  {nid}{l}{_q(text)}{r}")
        return nid

    def edge(self, a: str, b: str, label: str = "", dotted: bool = False) -> None:
        arrow = "-.->" if dotted else "-->"
        self.lines.append(f"  {a} {arrow}{'|' + _q(label)[1:-1] + '|' if label else ''} {b}")

    def steps(self, steps: list, prev: str) -> str:
        for s in steps:
            prev = self.step(s, prev)
        return prev

    def step(self, s: dict, prev: str) -> str:
        k, v = next(iter(s.items()))
        if k == "wait":
            n = self.node(v.get("label") or f"Wait {float(v['seconds']):g} s")
        elif k in ("wait_until", "expect"):
            n = self.node(f"{KIND_WORD[k]}: {v['label']} (up to {float(v['timeout_s']):g} s)",
                          "expect" if k == "expect" else "box")
        elif k == "fire":
            label = v.get("label") or v["card"]
            n = self.node(f"{label} [{v['card']}]", "event")
        elif k == "parallel":
            split = self.node(f"At the same time: {v['label']}", "split")
            self.edge(prev, split)
            join = None
            ends = [self.steps(b, split) for b in v["branches"]]
            join = self.node("All done", "end")
            for e in ends:
                self.edge(e, join)
            return join
        elif k == "unless":
            d = self.node(f"Unless: {v['label']}", "decide")
            self.edge(prev, d)
            end = self.steps(v["then"], d)
            join = self.node("Go on", "end")
            self.edge(end, join)
            self.edge(d, join, "it happens first", dotted=True)
            return join
        else:
            raise ValueError(k)
        self.edge(prev, n)
        return n


def mermaid(flow: Flow) -> str:
    m = _Mermaid()
    start = m.node("Run starts", "end")
    m.steps(flow.steps, start)
    return "\n".join(m.lines)


def markdown(flows: list[Flow]) -> str:
    out = ["# Scenario flows", "",
           "Each scenario preset has a flow: what should happen, in order. The director runs it as a py_trees "
           "behaviour tree, every 0.5 s of simulated time. A step that waits for something (the robot greets someone, "
           "a person sits down) holds the flow until it happens or until its time limit, so a scenario always moves "
           "on. An event step plays a storyboard card. An expect step is what the robot should do in reply: the run "
           "records whether it did.",
           "",
           "The flows are YAML files in `config/presets/flows/`; the format is in `wardzero_agents/flows/spec.py`. "
           "When a user rewrites a scenario's prompt in the Studio, the flow is adapted to the new prompt by a model "
           "before the next run starts (`wardzero_agents/flows/generate.py`), checked against the scenario's people, "
           "cards and rooms, and shown under View flow. Removing a person a flow refers to drops the events aimed at "
           "them.",
           "",
           "This file is made by `python scripts/flows_doc.py`; a test fails when it is out of date.", ""]
    for f in flows:
        if f.id in DOC_SKIP:
            continue
        out += [f"## {f.title} (`{f.id}`)", "", f.summary.strip(), "", "```mermaid", mermaid(f), "```", ""]
    return "\n".join(out)
