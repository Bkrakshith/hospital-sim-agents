#!/usr/bin/env python3
"""Rebuild config/isaac/characters.yaml, the allow-list of Isaac People characters, from the asset metadata.

    python scripts/curate_characters.py [--assets ~/isaacsim_assets/Assets/Isaac/6.0] [--variants 3]

Plain Python (no Isaac). Rules, from the brief: no hi-vis, hard hats or workwear.

- Characters/: only the scrubs (Medical) and the business suit are kept (police and construction are out).
- DH_Characters(_Extended): the metadata json names the outfit. WORKWEAR is out. A CASUAL outfit stays, and the
  character's whole ``Accessories`` prim (safety helmet, ear defenders, gloves, glasses, mask) is hidden at spawn.
  Per body, the base character and up to ``--variants`` Extended re-skins are kept.
The file is the curated result; edit it by hand if the contact sheet (scripts/character_sheet.py) shows a bad one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

PEOPLE = "Isaac/People"
SCRUBS = {"M_Medical_01": "male", "F_Medical_01": "female", "original_male_adult_medical_01": "male",
          "original_female_adult_medical_01": "female"}
CIVILIAN_GENERIC = {"F_Business_02": "female", "original_female_adult_business_02": "female"}
OLDER_FROM = 60
HEADER = """# Allow-list of Isaac People characters for the Isaac host (I2). Built by scripts/curate_characters.py.
# Excluded on purpose: hi-vis, hard hats and workwear (construction, police, the DH WORKWEAR outfits).
# usd is relative to the Isaac assets root; hide lists prims under the SkelRoot that are deactivated at spawn.
# outfit: civilian (casual clothes), scrubs (medical). age: young_adult, adult or older (60+).
"""


def band(age: int) -> str:
    return "older" if age >= OLDER_FROM else ("young_adult" if age < 30 else "adult")


def build(assets: Path, variants: int) -> dict:
    people = assets / PEOPLE
    chars: list[dict] = []
    for name, sex in {**SCRUBS, **CIVILIAN_GENERIC}.items():
        found = sorted((people / "Characters" / name).glob("*.usd"))  # the original_* folders name the file differently
        usd = found[0] if found else None
        if usd is not None:
            scrubs = name in SCRUBS
            chars.append({"id": name, "usd": usd.relative_to(assets).as_posix(), "sex": sex, "age": "adult",
                          "age_years": 35, "outfit": "scrubs" if scrubs else "civilian", "hide": []})
    kept: dict[str, int] = {}
    for folder in ("DH_Characters", "DH_Characters_Extended"):
        for js in sorted((people / folder).glob("*/*.json")):
            meta = json.loads(js.read_text())
            usd = js.with_suffix(".usd")
            if "gender" not in meta or not usd.exists() or "CASUAL" not in (meta.get("outfit") or ""):
                continue
            body = meta["id"]
            is_base = folder == "DH_Characters"
            if not is_base and kept.get(body, 0) > variants:
                continue
            kept[body] = kept.get(body, 0) + 1
            chars.append({"id": "dh_" + usd.stem[:8] + usd.stem[36:], "usd": usd.relative_to(assets).as_posix(),
                          "sex": meta["gender"], "age": band(int(meta["age"])), "age_years": int(meta["age"]),
                          "outfit": "civilian", "hide": ["Accessories"]})
    seen, unique = set(), []
    for c in chars:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)
    return {"version": 1, "characters": unique}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(Path.home() / "isaacsim_assets/Assets/Isaac/6.0"))
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "config/isaac/characters.yaml"))
    a = ap.parse_args()
    data = build(Path(a.assets), a.variants)
    Path(a.out).write_text(HEADER + yaml.safe_dump(data, sort_keys=False, width=140))
    print(f"{len(data['characters'])} characters -> {a.out}")


if __name__ == "__main__":
    main()
