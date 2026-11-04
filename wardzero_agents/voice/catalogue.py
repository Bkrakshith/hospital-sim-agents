"""The voices WardZero can speak with: which Piper models, their licences, and which person each one suits.

Models are downloaded by ``scripts/fetch_voices.py`` into ``<data>/voices`` (never committed). All voices are British
English (the site is a UK hospital). Licences were read from each model card on 2026-11-04; every one allows generated
audio to be redistributed (CC BY 4.0 and public domain need credit or nothing; CC BY-SA 4.0 asks that adaptations of the
dataset stay share-alike and carry credit, which the NOTICE of a published film should give).
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Optional

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB"
AGE_BANDS = ("young_adult", "adult", "older")
DEFAULT_ROBOT_VOICE = "en_GB-cori-medium"


@dataclasses.dataclass(frozen=True)
class Model:
    """One downloadable Piper model: files and hashes pinned, plus its licence."""
    stem: str  # en_GB-cori-medium
    path: str  # under HF_BASE, without extension: cori/medium/en_GB-cori-medium
    licence: str
    credit: str
    sha256_onnx: str
    sha256_json: str


MODELS = {m.stem: m for m in (
    Model("en_GB-cori-medium", "cori/medium/en_GB-cori-medium", "Public domain (LibriVox recordings)",
          "Bryce Beattie, from LibriVox readings",
          "1899f98e5fb8310154f3c2973f4b8a929ba7245e722b3d3a85680b833d95f10d", "e262c16d7f192f69d4edd6b4ef8a5915379e67495fcc402f1ab15eeb33da3d36"),
    Model("en_GB-alba-medium", "alba/medium/en_GB-alba-medium", "CC BY 4.0",
          "Alba voice, CSTR / University of Edinburgh (datashare 10283/3270)",
          "401369c4a81d09fdd86c32c5c864440811dbdcc66466cde2d64f7133a66ad03b", "aa965a2f02ecced632c2694e1fc72bbff6d65f265fab567ca945918c73dd89f4"),
    Model("en_GB-northern_english_male-medium", "northern_english_male/medium/en_GB-northern_english_male-medium",
          "CC BY-SA 4.0", "Google crowdsourced UK/Ireland English (openslr.org/83)",
          "57a219ae8e638873db7d18893304be5069c42868f392bb95c3ff17f0690d0689", "69557ed3d974463453e9b0c09dd99a7ed0e52b8b87b64b357dbeeb2540a97d47"),
    Model("en_GB-vctk-medium", "vctk/medium/en_GB-vctk-medium", "CC BY 4.0",
          "CSTR VCTK corpus, University of Edinburgh (datashare 10283/3443)",
          "4e9fc85ab9009385319fc6bae7f55577f8a2d7ee77fd9159a5500eb6531f41e6", "7f85e6391ed0f7f46e4abd19345929a16be931a0c9945086f96692dce2087fa8"),
)}


@dataclasses.dataclass(frozen=True)
class Voice:
    key: str  # unique: the model stem, or stem + "#" + speaker for a multi-speaker model
    model: str  # a MODELS stem
    sex: str  # female | male
    ages: tuple  # age bands it suits, the closest first
    speaker: Optional[str] = None  # speaker name inside a multi-speaker model
    length_scale: float = 1.0  # above 1 is slower

    @property
    def licence(self) -> str:
        return MODELS[self.model].licence


# VCTK speaker -> (sex, age bands). Sex was checked by the median pitch of each speaker's synthesis (men 80 to 150 Hz,
# women 155 to 256 Hz); ages are the corpus's (18 to 38, and one speaker of 52), so most are young adults.
_VCTK_MEN = ("p254 p326 p237 p284 p256 p245 p347 p281 p334 p345 p360 p287 p316 p274 p311 p364 p304 p263 p376 p246 p260 "
             "p270 p275 p292 p363 p278 p226 p232 p374 p271 p241 p259 p285 p243 p252 p298 p227 p258 p272 p279 p251 p302 "
             "p247 p286 p255 p273").split()
_VCTK_WOMEN = ("p343 p262 p299 p301 p294 p225 p313 p249 p267 p280 p341 p231 p314 p229 p297 p306 p333 p282 p234 p269 "
               "p288 p308 p265 p266 p330 p318 p239 p295 p361 p293 p230 p300 p228 p238 p244 p264 p277 p323 p250 p339 "
               "p257 p335 p233 p312 p268 p283 p329 p336 p340 p276 p362 p303 p351 p253 p236 p240 p248 p305 p261 p310 "
               "p307 p317").split()
_VCTK_ADULT = {"p227", "p294", "p318", "p281", "p362", "p374"}  # 28 to 38 years old
_VCTK_OLDER = {"p280"}  # 52


def _vctk(name: str, sex: str) -> Voice:
    ages = ("older", "adult", "young_adult") if name in _VCTK_OLDER else (
        ("adult", "young_adult", "older") if name in _VCTK_ADULT else ("young_adult", "adult", "older"))
    return Voice(f"en_GB-vctk-medium#{name}", "en_GB-vctk-medium", sex, ages, name,
                 1.08 if name in _VCTK_OLDER else 1.0)


VOICES = {v.key: v for v in (
    Voice("en_GB-cori-medium", "en_GB-cori-medium", "female", ("adult", "older", "young_adult")),
    Voice("en_GB-alba-medium", "en_GB-alba-medium", "female", ("older", "adult", "young_adult"), length_scale=1.06),
    Voice("en_GB-northern_english_male-medium", "en_GB-northern_english_male-medium", "male",
          ("older", "adult", "young_adult"), length_scale=1.06),
    *(_vctk(n, "male") for n in _VCTK_MEN), *(_vctk(n, "female") for n in _VCTK_WOMEN),
)}


def voices_dir(data: Optional[Path] = None) -> Path:
    """$WARDZERO_VOICES_DIR, else <data>/voices."""
    env = os.environ.get("WARDZERO_VOICES_DIR")
    if data is None and env:
        return Path(env).expanduser()
    if data is None:
        from ..memory import data_dir
        data = data_dir()
    return Path(data) / "voices"


def installed_models(directory: Optional[Path] = None) -> set[str]:
    """Stems whose .onnx and .onnx.json are both in the folder."""
    d = voices_dir() if directory is None else Path(directory)
    return {s for s in MODELS if (d / f"{s}.onnx").is_file() and (d / f"{s}.onnx.json").is_file()}
