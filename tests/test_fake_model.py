"""The scripted (fake) model: people who can take part in every preset's main task, a debrief that can pass, and a
flow adapter whose change shows."""
import asyncio
from pathlib import Path

import pytest

from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.llm.fake import FakeGateway, default_policy
from wardzero_agents.llm.gateway import LLMRequest
from wardzero_agents.mind.prompts import decision_schema, persona_card, robot_role_of

CONFIG = Path(__file__).resolve().parents[1] / "config"
PRESETS = load_scenario_presets(CONFIG)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    out = {}

    def get(pid):
        if pid not in out:
            out[pid] = preset_to_scenario(PRESETS[pid], CONFIG, tmp_path_factory.mktemp(pid))
        return out[pid]
    return get


def ask(loaded, pid, situation, history=()):
    """What the scripted model decides for person pid, given the situation lines (and the day so far)."""
    inst = next(i for i in loaded.instances if i.person_id == pid)
    card = persona_card(inst, loaded.area, loaded.site, loaded.scenario.known_places,
                        robot_role=robot_role_of(loaded.scenario))
    hist = ["What has happened so far today (oldest first):", *[f"- {h}" for h in history]] if history else []
    req = LLMRequest(role="decide", person=pid, system_fixed="", persona_card=card,
                     user="Decide what you do next. The situation now:\n" + "\n".join(situation),
                     schema=decision_schema(inst.role), history=hist)
    return default_policy(req)


def kind(d):
    return d["intent"]["kind"]


# ---- hand-overs -------------------------------------------------------------------------------------------------------
TRAY_OFFER = ("NUDGE: The robot is holding out the tray ({label}) for you. You are lying flat: if you want it, accept "
              "it (accept_item) and you sit up in bed to take it, with no separate sit_up_in_bed needed; if not, refuse "
              "it (refuse_item).")


def test_a_patient_accepts_her_own_tray_and_refuses_someone_elses(staged):
    loaded = staged("meal_round")
    own = ask(loaded, "M1", [TRAY_OFFER.format(label="Margaret Hall - soft diet")])
    other = ask(loaded, "M3", [TRAY_OFFER.format(label="Margaret Hall - soft diet")])
    assert kind(own) == "accept_item"
    assert kind(other) == "refuse_item"


def test_a_sleeping_patient_does_not_answer(staged):
    loaded = staged("meal_round")
    asleep = "I felt: You are deeply asleep. Do not react unless someone speaks to you loudly or touches your arm."
    d = ask(loaded, "M2", [TRAY_OFFER.format(label="George Wright - diabetic")], history=[f"t=10s; {asleep}"])
    assert kind(d) == "sit_idle" and not d["say"]
    # spoken to by name, she wakes and can take it
    woke = ask(loaded, "M2", [TRAY_OFFER.format(label="George Wright - diabetic")],
               history=[f"t=10s; {asleep}", 't=40s; the robot said to me "Hello George, here is your lunch"'])
    assert kind(woke) == "accept_item"


def test_the_porter_takes_the_specimen_box_and_others_refuse_it(staged):
    loaded = staged("specimen_run")
    offer = ("NUDGE: The robot is holding out the specimen box (blood specimens, ward bay 2) to you. If you want it, "
             "take it (accept_item); if not, refuse it (refuse_item).")
    assert kind(ask(loaded, "PO1", [offer])) == "accept_item"
    assert kind(ask(loaded, "V1", [offer])) != "accept_item"


# ---- the surgeon --------------------------------------------------------------------------------------------------------
BATCH = ("Over the next few minutes ask for these instruments, one at a time and in this order, by name: scalpel, "
         "forceps, retractor. Ask more sharply if one is late. Reject a wrong one and say so.")


def test_the_surgeon_names_each_instrument_in_turn_and_accepts_only_that_one(staged):
    loaded = staged("scrub_nurse")
    first = ask(loaded, "S1", [f"NUDGE: {BATCH}"])
    assert kind(first) == "request_instrument" and first["intent"]["text"] == "scalpel"
    assert first["addressee"] == "g1"
    asked = [f"t=30s; I felt: {BATCH}; I chose request_instrument (mood calm)"]
    offer = "NUDGE: The robot is holding out the {} to you. If you want it, take it (accept_item); if not, refuse it " \
            "(reject_item)."
    assert kind(ask(loaded, "S1", [offer.format("forceps")], asked)) == "reject_item"
    assert kind(ask(loaded, "S1", [offer.format("scalpel")], asked)) == "accept_item"
    waiting = ask(loaded, "S1", ["Why you are deciding now: robot_near"], asked)
    assert kind(waiting) == "wait_standing"  # asked already: no second request, and no sit_idle for a surgeon
    done = asked + ["t=34s; I chose accept_item (mood calm)"]
    nxt = ask(loaded, "S1", ["Why you are deciding now: intent_done"], done)
    assert kind(nxt) == "request_instrument" and nxt["intent"]["text"] == "forceps"


def test_the_surgeon_idles_in_a_way_her_role_allows(staged):
    loaded = staged("scrub_nurse")
    d = ask(loaded, "S1", ["Why you are deciding now: start"])
    assert kind(d) in decision_schema("surgeon")["properties"]["intent"]["properties"]["kind"]["enum"]


# ---- asking the robot for help ------------------------------------------------------------------------------------------
def test_someone_who_wants_help_asks_the_robot_in_so_many_words(staged):
    loaded = staged("reception_guide")
    d = ask(loaded, "V1", ["NUDGE: You can't work out where the ward is from the signs. Find someone who can help."])
    assert kind(d) == "approach" and d["intent"]["target"] == "g1"
    assert d["addressee"] == "g1" and d["say"].startswith("Excuse me, can you help") and "ward" in d["say"]


def test_spoken_directions_to_a_place_on_the_card_are_followed(staged):
    loaded = staged("evacuation_guide")
    d = ask(loaded, "V1", ["ROBOT SAID TO YOU: The main entrance is blocked. Please walk to the Ambulance handover "
                           "bay now."])
    assert kind(d) == "walk_to" and d["intent"]["target"] == "AE_Handover"


def test_a_walk_in_progress_carries_on(staged):
    loaded = staged("evacuation_guide")
    d = ask(loaded, "V1", ["Current activity: walk_to AE_Handover (running).", "Why you are deciding now: speech_nearby"])
    assert kind(d) == "walk_to" and d["intent"]["target"] == "AE_Handover"


def test_a_nurse_called_by_the_robot_goes_to_the_patient(staged):
    loaded = staged("night_call_bell")
    d = ask(loaded, "N1", ["NUDGE: The robot pressed the staff call from Ward bedroom B2; call light on at bed B2. The "
                           "person who needs a nurse is person id P1 (they have not said their name), in Ward bedroom "
                           "B2 at x 25.1, y 19.1: go to them (approach P1), help, then go back to your work."])
    assert kind(d) == "approach" and d["intent"]["target"] == "P1"


# ---- debrief and flow adapter -------------------------------------------------------------------------------------------
def test_the_scripted_debrief_rates_above_the_presets_thresholds():
    req = LLMRequest(role="summarise", person="P1", system_fixed="", persona_card="", user="",
                     schema={"properties": {"likeability": {}, "safety": {}, "competence": {}, "trust": {}}})
    d = default_policy(req)
    assert min(d[k] for k in ("likeability", "safety", "competence", "trust")) >= 4
    thresholds = [c.threshold for p in PRESETS.values() for c in p.criteria if c.metric.startswith("rating_")]
    assert thresholds and all(4 >= th for th in thresholds)


def test_the_scripted_adapter_changes_the_flow_visibly():
    from wardzero_agents.config.loader import _load_deck
    from wardzero_agents.flows import generate as G
    from wardzero_agents.flows.spec import flows_dir, load_flow
    from wardzero_agents.site import load_site
    p = PRESETS["ae_waiting"]
    base = load_flow(flows_dir(CONFIG) / "ae_waiting.yaml")
    people = [{"id": c.id, "name": c.name, "role": c.role, "room": c.room} for c in p.characters]
    prompt = "George arrives with a bad cough and Ava is very anxious."
    flow, notes = asyncio.run(G.adapt_flow(FakeGateway(), scenario_id="ae_waiting", base=base, preset_prompt=p.prompt,
                                           prompt=prompt, people=people, deck=_load_deck(CONFIG / "cards" / "studio.yaml"),
                                           rooms=load_site(CONFIG).rooms, duration_min=p.duration_min))
    assert flow.steps == base.steps  # still the preset's steps, so the run plays the same scenario
    assert flow.summary != base.summary and prompt in flow.summary
    assert "kept" in notes and flow.prompt_sha == G.prompt_sha(prompt)
