"""Role presets: which intents each role may choose (spec 8.1)."""
from __future__ import annotations

BASIC_INTENTS: frozenset[str] = frozenset({
    "sit_idle", "use_phone", "fidget", "look_at", "read_sign", "stand_up", "sit_down", "walk_to",
    "approach", "wait_standing", "talk", "gesture", "cough", "leave",
})

_STAFF_EXTRA = {"attend", "give_item", "take_item", "call_for_help", "escort", "evacuate",
                "use_extinguisher"}
_BED = {"sit_up_in_bed", "lie_back", "call_bell"}

# A surgeon is limited to the table: no walking, sitting or leaving.
_SURGEON = {"talk", "look_at", "gesture", "wait_standing", "fidget", "cough", "request_instrument",
            "accept_item", "reject_item", "take_instrument_self"}

ROLE_INTENTS: dict[str, frozenset[str]] = {
    "ward_nurse": frozenset(BASIC_INTENTS | _STAFF_EXTRA),
    "staff": frozenset(BASIC_INTENTS | _STAFF_EXTRA),
    "receptionist": frozenset(BASIC_INTENTS | {"call_for_help"}),
    "patient": frozenset(BASIC_INTENTS | _BED | {"accept_item", "refuse_item"}),
    "visitor": frozenset(BASIC_INTENTS | {"escort", "call_for_help"}),
    "companion": frozenset(BASIC_INTENTS | {"escort", "call_for_help"}),
    "surgeon": frozenset(_SURGEON),
    "porter": frozenset(BASIC_INTENTS | {"take_item", "give_item", "accept_item", "refuse_item",
                                         "call_for_help"}),
    "custom": frozenset(BASIC_INTENTS),
}
