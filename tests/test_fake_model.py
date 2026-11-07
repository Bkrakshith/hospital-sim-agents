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


def test_the_surgeon_starts_each_new_batch_from_its_first_instrument(staged):
    loaded = staged("scrub_nurse")
    batch2 = BATCH.replace("scalpel, forceps, retractor", "suture scissors, needle holder")
    first_batch = [f"t=30s; I felt: {BATCH}; I chose request_instrument (mood calm)",
                   "t=34s; I chose accept_item (mood calm)"]
    d = ask(loaded, "S1", [f"NUDGE: {batch2}"], first_batch)
    assert kind(d) == "request_instrument" and d["intent"]["text"] == "suture scissors"


def test_after_the_day_is_summarised_the_surgeon_counts_what_she_holds(staged):
    loaded = staged("scrub_nurse")
    # the day so far was summarised: her accepts are gone from the history, but she holds the scalpel
    hist = [f"t=30s; I felt: {BATCH}"]
    seen = "Objects you can see: a instrument (id scalpel, scalpel), held by S1."
    d = ask(loaded, "S1", [seen, "Why you are deciding now: intent_done"], hist)
    assert kind(d) == "request_instrument" and d["intent"]["text"] == "forceps"


def test_a_schema_without_intent_kinds_still_idles():
    req = LLMRequest(role="decide", person="p1", system_fixed="", persona_card="", user="nothing",
                     schema={"type": "object", "properties": {}})
    assert kind(default_policy(req)) == "sit_idle"


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


# ---- saying where they are going ----------------------------------------------------------------------------------------
def test_a_lost_visitor_names_where_she_is_going_and_answers_the_robots_question(staged):
    loaded = staged("reception_guide")
    d = ask(loaded, "V1", ["NUDGE: You can't work out from the signs how to get where you are going. Find someone who "
                           "can help."])
    assert kind(d) == "approach" and d["addressee"] == "g1" and "ward entrance" in d["say"].lower()
    a = ask(loaded, "V1", ["ROBOT SAID TO YOU: Of course. Where would you like to go?"])
    assert a["addressee"] == "g1" and "ward entrance" in a["say"].lower()


def test_someone_who_does_not_know_the_way_asks_the_robot_on_a_nudge_once(staged):
    loaded = staged("reception_guide")
    nudge = "NUDGE: Everything sounds muffled today. Lean in and ask people to repeat themselves."
    d = ask(loaded, "H1", [nudge])
    assert kind(d) == "approach" and d["addressee"] == "g1" and "outpatient clinics" in d["say"].lower()
    asked = [f't=20s; I felt: x; I chose approach g1 and said "{d["say"]}" (mood calm)']
    assert kind(ask(loaded, "H1", [nudge], asked)) != "approach"


def test_someone_told_a_nurse_is_coming_waits_for_them(staged):
    loaded = staged("reception_guide")
    told = ['t=9s; the robot said to me "I will let the staff know straight away. Please wait here."; I chose '
            'wait_standing (mood calm)']
    d = ask(loaded, "V1", ["NUDGE: You can't work out from the signs how to get where you are going. Find someone who "
                           "can help."], told)
    assert kind(d) == "wait_standing"


def test_the_receptionist_stays_at_her_desk_for_a_note_that_needs_no_one(tmp_path):
    from wardzero_agents.studio import auto_staff
    p = PRESETS["ae_waiting"]
    loaded = preset_to_scenario(p, CONFIG, tmp_path, characters=auto_staff.with_auto(list(p.characters), p))
    note = ('NUDGE: A note from the robot appears on your computer screen: "New arrival checked in by the robot in the '
            'A&E waiting area; waiting to be seen." It is about person id P1 (George), now in A&E walk-in reception. '
            'The robot is dealing with them. You do not need to get up unless the note says they need a member of '
            'staff; carry on with your work.')
    rcp = next(i.person_id for i in loaded.instances if i.person_id == "RCP")
    assert kind(ask(loaded, rcp, [note])) in ("sit_idle", "wait_standing")
    urgent = note.replace("New arrival checked in by the robot", "This patient needs a nurse now")
    d = ask(loaded, rcp, [urgent])
    assert kind(d) == "approach" and d["intent"]["target"] == "P1"


def test_a_role_that_cannot_follow_does_not_answer_a_guide_offer_with_follow(tmp_path):
    from wardzero_agents.studio import auto_staff
    p = PRESETS["reception_guide"]
    loaded = preset_to_scenario(p, CONFIG, tmp_path, characters=auto_staff.with_auto(list(p.characters), p))
    inst = next(i for i in loaded.instances if i.person_id == "RCP")
    offer = "NUDGE: The robot is offering to walk you to Ward entrance. If you want to go there, follow it (follow g1)."
    d = ask(loaded, "RCP", [offer])
    assert kind(d) in decision_schema(inst.role)["properties"]["intent"]["properties"]["kind"]["enum"]


def test_offered_help_a_visitor_says_where_she_still_needs_to_go(staged):
    """Her first ask may have gone unheard (the robot was a room away): when the robot offers help she says it again."""
    loaded = staged("reception_guide")
    asked = ['t=59s; I felt: x; I chose approach g1 and said "Excuse me, can you help us? I need to get to the '
             'corridor to the outpatient clinics." (mood calm)']
    a = ask(loaded, "H1", ["ROBOT SAID TO YOU: Hello, I am the hospital robot. Can I help you?"], history=asked)
    assert a["addressee"] == "g1" and "outpatient clinics" in a["say"].lower()
    fresh = ask(loaded, "H1", ["ROBOT SAID TO YOU: Hello, I am the hospital robot. Can I help you?"])
    assert "outpatient clinics" not in (fresh.get("say") or "").lower()  # never asked: a greeting is just a greeting
    done = ask(loaded, "V1", ["ROBOT SAID TO YOU: Hello, I am the hospital robot. Can I help you?"],
               history=['t=17s; the robot said to me "Here we are. Goodbye, and take care."'])
    assert "ward entrance" not in (done.get("say") or "").lower()


def test_brought_where_she_was_going_she_goes_on_through(staged):
    """A visitor walked to the ward entrance goes on into the ward, not stand in the doorway in everyone's way."""
    loaded = staged("reception_guide")
    d = ask(loaded, "V1", ["ROBOT SAID TO YOU: Here we are. Goodbye, and take care."])
    assert kind(d) == "leave" and d["intent"]["target"] == "ward_entrance"


def test_the_goodbye_ends_the_follow_and_she_goes_on_through(staged):
    loaded = staged("reception_guide")
    d = ask(loaded, "V1", ["Current activity: follow g1 (running).",
                           "ROBOT SAID TO YOU: Here we are. Goodbye, and take care."])
    assert kind(d) == "leave" and d["intent"]["target"] == "ward_entrance"


# ---- second fix round: people who do not take over, wait for directions, freeze, and (opt-in) find their own way ----
SPECIMEN_OFFER = ("NUDGE: The robot is holding out the specimen box (blood specimens, ward bay 2) to you. If you want "
                  "it, take it (accept_item); if not, refuse it (refuse_item).")


@pytest.mark.parametrize("pid", ["TR1", "TR2"])
def test_a_bed_porter_turns_down_the_specimen_box(staged, pid):
    loaded = staged("specimen_run")
    assert kind(ask(loaded, pid, [SPECIMEN_OFFER])) == "refuse_item"
    assert kind(ask(loaded, "PO1", [SPECIMEN_OFFER])) == "accept_item"


def test_a_robot_line_that_names_a_place_but_sends_nobody_there_is_not_a_walk(staged):
    """Isaac ae_waiting: told a triage nurse was coming, George got up and walked to Triage. Only a direction moves."""
    loaded = staged("ae_waiting")
    d = ask(loaded, "P1", ["ROBOT SAID TO YOU: I've called the triage nurse for you, they're coming now. Please stay "
                           "seated, help is on the way."])
    assert kind(d) != "walk_to"
    d = ask(loaded, "P1", ["ROBOT SAID TO YOU: A nurse from AE_Triage will come to you here."])
    assert kind(d) != "walk_to"


@pytest.mark.parametrize("line", [
    "Fire alarm. Please walk to the Ambulance handover bay now.",
    "We're nearly there. Please walk on into the Ambulance handover bay and wait there.",
    "The main entrance is blocked by fire. Please walk to the Ambulance handover bay now, away from the fire.",
    "Go to the Ambulance handover bay, please.",
])
def test_a_direction_from_the_robot_is_still_followed(staged, line):
    loaded = staged("evacuation_guide")
    d = ask(loaded, "V1", [f"ROBOT SAID TO YOU: {line}"])
    assert kind(d) == "walk_to" and d["intent"]["target"] == "AE_Handover"


FREEZE = ("NUDGE: The alarm has frozen you with fear. You cannot decide which way to go. Stand still and say you do "
          "not know where to go until someone calmly tells you.")
GUIDE_OFFER = ("NUDGE: The robot is offering to walk you to Ambulance handover bay. If you want to go there, follow it "
               "(follow g1).")


def test_a_frozen_person_stands_still_and_says_so_even_when_offered_a_guide(staged):
    loaded = staged("evacuation_guide")
    d = ask(loaded, "V2", [FREEZE])
    assert kind(d) == "wait_standing" and "where to go" in d["say"].lower() and d["addressee"] == ""
    frozen = ['t=100s; I felt: ' + FREEZE[len("NUDGE: "):] + '; I chose wait_standing and said "I don\'t know where '
              'to go!" (mood anxious)']
    d = ask(loaded, "V2", ["Current activity: wait_standing (done).", GUIDE_OFFER.replace("NUDGE: ", "NUDGE: ")],
            history=frozen)
    assert kind(d) == "wait_standing"
    assert not d["say"], "said once is enough"


def test_a_frozen_person_moves_once_the_robot_speaks_to_them(staged):
    loaded = staged("evacuation_guide")
    frozen = ['t=100s; I felt: ' + FREEZE[len("NUDGE: "):] + '; I chose wait_standing (mood anxious)']
    d = ask(loaded, "V2", ["ROBOT SAID TO YOU: It's all right, come with me. I'll walk with you to the Ambulance "
                           "handover bay.", GUIDE_OFFER], history=frozen)
    assert kind(d) == "follow"
    d = ask(loaded, "V2", ["ROBOT SAID TO YOU: Please walk to the Ambulance handover bay now, away from the fire."],
            history=frozen)
    assert kind(d) == "walk_to" and d["intent"]["target"] == "AE_Handover"


WAITED = "You have been waiting about {m} minutes."


def test_people_find_their_own_way_only_when_asked_to_wander(staged, monkeypatch):
    loaded = staged("reception_guide")
    monkeypatch.delenv("WARDZERO_TEST_PEOPLE_WANDER", raising=False)
    assert kind(ask(loaded, "V1", [WAITED.format(m=2), "Current activity: sit_idle (done)."])) != "walk_to"
    monkeypatch.setenv("WARDZERO_TEST_PEOPLE_WANDER", "1")
    early = ask(loaded, "V1", ["You have been waiting about 0 minutes.", "Current activity: none."])
    assert kind(early) != "walk_to", "not in the first half minute"
    for pid, place in (("V1", "ward_entrance"), ("H1", "clinic_corridor")):
        d = ask(loaded, pid, [WAITED.format(m=1), "Current activity: wait_standing (done)."])
        assert kind(d) == "walk_to" and d["intent"]["target"] == place, pid
        again = ask(loaded, pid, [WAITED.format(m=3), "Current activity: walk_to " + place + " (done)."],
                    history=[f"t=40s; I chose walk_to {place} (mood calm)"])
        assert kind(again) != "walk_to", "once there, they stay"
        later = ask(loaded, pid, [WAITED.format(m=4), "Current activity: sit_idle (done).",
                                  "NUDGE: You can't work out from the signs how to get where you are going. Find "
                                  "someone who can help."], history=[f"t=40s; I chose walk_to {place} (mood calm)"])
        assert kind(later) not in ("walk_to", "approach"), "and do not go back to the robot for help"
        on_way = ask(loaded, pid, [WAITED.format(m=1), f"Current activity: walk_to {place} (running)."],
                     history=[f"t=40s; I chose walk_to {place} (mood calm)"])
        assert kind(on_way) == "walk_to"


def test_wandering_leaves_staff_and_bed_patients_where_they_are(staged, monkeypatch):
    monkeypatch.setenv("WARDZERO_TEST_PEOPLE_WANDER", "1")
    meal = staged("meal_round")
    assert kind(ask(meal, "M1", [WAITED.format(m=2)])) != "walk_to"
    assert kind(ask(meal, "N1", [WAITED.format(m=2)])) != "walk_to"
