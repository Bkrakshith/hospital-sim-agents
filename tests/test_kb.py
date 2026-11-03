import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.kb.chunker import Chunk, chunk_markdown
from wardzero_agents.kb.embed import FastEmbedder, HashEmbedder, get_embedder
from wardzero_agents.kb.index import KnowledgeBase, build_default_kb

AGENTS_ROOT = Path(__file__).resolve().parents[1]
HG1_ROOT = Path(os.environ.get("HG1_ROOT", AGENTS_ROOT.parent))
V1 = AGENTS_ROOT / "config" / "scenarios" / "waiting_area_v1.yaml"


def write(tmp_path, text, name="n.md"):
    p = tmp_path / name
    p.write_text(text)
    return p


def chunks_of(texts):
    return [Chunk(id=f"f#{i}", file="f", heading_path=[], text=t, links=[]) for i, t in enumerate(texts)]


def make_kb():
    return KnowledgeBase.build(
        chunks_of(["outpatients wait 15-45 minutes seated", "dialysis three times a week"]), HashEmbedder()
    )


def test_chunks_keep_headings_and_links(tmp_path):
    c = chunk_markdown(write(tmp_path, "# A\n## B\ntext https://x.org/a\n"), tmp_path)[0]
    assert c.heading_path == ["A", "B"] and c.links == ["https://x.org/a"]
    assert c.id == "n.md#0" and c.file == "n.md"


def test_chunk_paths_posix_relative(tmp_path):
    sub = tmp_path / "dir with space"
    sub.mkdir()
    c = chunk_markdown(write(sub, "# A\nbody\n"), tmp_path)[0]
    assert c.file == "dir with space/n.md"


def test_oversized_section_split_at_paragraphs(tmp_path):
    md = "# A\n" + "\n\n".join(f"para {i} " + "x" * 100 for i in range(10)) + "\n"
    cs = chunk_markdown(write(tmp_path, md), tmp_path, max_chars=300)
    assert len(cs) > 1
    assert all(len(c.text) <= 300 for c in cs)
    assert all(c.heading_path == ["A"] for c in cs)
    assert [c.id for c in cs] == [f"n.md#{i}" for i in range(len(cs))]


def test_heading_path_resets_on_sibling(tmp_path):
    cs = chunk_markdown(write(tmp_path, "# A\n## B\none\n## C\ntwo\n### D\nthree\n"), tmp_path)
    assert [c.heading_path for c in cs] == [["A", "B"], ["A", "C"], ["A", "C", "D"]]


def test_hash_embedder_normalised():
    v = HashEmbedder().embed(["cough in waiting room"])
    assert v.shape == (1, 384) and v.dtype == np.float32
    assert abs(np.linalg.norm(v[0]) - 1) < 1e-5


def test_hash_embedder_empty_text_is_zero_not_nan():
    v = HashEmbedder().embed([""])
    assert not np.isnan(v).any() and np.count_nonzero(v) == 0


def test_hash_embedder_deterministic_across_processes():
    code = (
        "from wardzero_agents.kb.embed import HashEmbedder;"
        "import sys;v=HashEmbedder().embed(['cough in waiting room'])[0];"
        "print(repr(v[:400].tolist()))"
    )
    outs = []
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True, cwd=AGENTS_ROOT).stdout)
    assert outs[0] == outs[1]
    assert outs[0].strip() == repr(HashEmbedder().embed(["cough in waiting room"])[0][:400].tolist())


def test_embedder_name_and_get_embedder():
    assert HashEmbedder().name == "hash384" and HashEmbedder(dim=128).name == "hash128"
    assert isinstance(get_embedder("hash"), HashEmbedder)
    with pytest.raises(ValueError):
        get_embedder("nope")


@pytest.mark.skipif(importlib.util.find_spec("fastembed") is not None, reason="fastembed installed")
def test_fast_embedder_without_package_gives_clear_error():
    with pytest.raises(RuntimeError, match=r"pip install -e \.\[embed\]"):
        FastEmbedder()


def test_search_finds_relevant():
    kb = make_kb()
    assert "wait" in kb.search("how long do outpatients wait", k=1)[0][0].text


def test_save_load_roundtrip(tmp_path):
    kb = make_kb()
    kb.save(tmp_path)
    kb2 = KnowledgeBase.load(tmp_path, HashEmbedder())
    assert kb2.search("dialysis", 1)[0][0].id == kb.search("dialysis", 1)[0][0].id
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert {"embedder", "dim", "built_at", "sources_sha256"} <= set(meta)
    assert (tmp_path / "chunks.jsonl").exists() and (tmp_path / "vectors.npy").exists()


def test_load_rejects_other_embedder(tmp_path):
    make_kb().save(tmp_path)
    with pytest.raises(ValueError, match="embedder"):
        KnowledgeBase.load(tmp_path, HashEmbedder(dim=128))


def test_default_kb_covers_research_notes():
    loaded = load_scenario(V1)
    kb = build_default_kb(HG1_ROOT, loaded, HashEmbedder())
    assert any("patient_personas.md" in c.file for c in kb.chunks)
    assert any(c.file == "<persona>" for c in kb.chunks)
    area = [c for c in kb.chunks if c.file == "<area>"]
    assert len(area) == 1
    assert next(iter(loaded.area.landmarks)) in area[0].text
    assert loaded.signs[0].text in area[0].text
    assert all("\\" not in c.file for c in kb.chunks)
    assert len(kb.sources_sha256) == 64


def test_sources_hash_changes_with_notes(tmp_path):
    notes = tmp_path / "research_notes"
    notes.mkdir()
    (notes / "a.md").write_text("# A\nhello\n")
    loaded = load_scenario(V1)
    h1 = build_default_kb(tmp_path, loaded, HashEmbedder()).sources_sha256
    (notes / "a.md").write_text("# A\nhello again\n")
    h2 = build_default_kb(tmp_path, loaded, HashEmbedder()).sources_sha256
    assert h1 != h2


def test_sources_hash_changes_with_landmark_label(tmp_path):
    (tmp_path / "research_notes").mkdir()
    loaded = load_scenario(V1)
    h1 = build_default_kb(tmp_path, loaded, HashEmbedder()).sources_sha256
    lid = next(iter(loaded.area.landmarks))
    loaded.area.landmarks[lid].label += " renamed"
    assert build_default_kb(tmp_path, loaded, HashEmbedder()).sources_sha256 != h1


def test_sources_hash_changes_with_max_chars(tmp_path):
    (tmp_path / "research_notes").mkdir()
    loaded = load_scenario(V1)
    h1 = build_default_kb(tmp_path, loaded, HashEmbedder()).sources_sha256
    assert build_default_kb(tmp_path, loaded, HashEmbedder(), max_chars=500).sources_sha256 != h1


def test_load_rejects_truncated_chunks(tmp_path):
    make_kb().save(tmp_path)
    lines = (tmp_path / "chunks.jsonl").read_text().splitlines()
    (tmp_path / "chunks.jsonl").write_text(lines[0] + "\n")
    with pytest.raises(ValueError, match="shape"):
        KnowledgeBase.load(tmp_path, HashEmbedder())
    (tmp_path / "chunks.jsonl").write_text(lines[0] + "\n" + lines[1][:10])
    with pytest.raises(ValueError, match="corrupt"):
        KnowledgeBase.load(tmp_path, HashEmbedder())


def test_load_rejects_wrong_shape_vectors(tmp_path):
    make_kb().save(tmp_path)
    np.save(tmp_path / "vectors.npy", np.zeros((2, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        KnowledgeBase.load(tmp_path, HashEmbedder())


def test_load_casts_vectors_to_float32(tmp_path):
    make_kb().save(tmp_path)
    v = np.load(tmp_path / "vectors.npy").astype(np.float64)
    np.save(tmp_path / "vectors.npy", v)
    assert KnowledgeBase.load(tmp_path, HashEmbedder()).vectors.dtype == np.float32


def test_save_is_atomic_and_overwrites(tmp_path):
    d = tmp_path / "kb"
    make_kb().save(d)
    make_kb().save(d)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["kb"]
    assert KnowledgeBase.load(d, HashEmbedder()).search("dialysis", 1)


def test_bom_file_keeps_first_heading(tmp_path):
    p = tmp_path / "n.md"
    p.write_bytes(b"\xef\xbb\xbf# A\nbody\n")
    assert chunk_markdown(p, tmp_path)[0].heading_path == ["A"]


def test_non_utf8_raises_value_error_naming_file(tmp_path):
    p = tmp_path / "bad.md"
    p.write_bytes(b"# A\n\xff\xfe\xfa body\n")
    with pytest.raises(ValueError, match="bad.md"):
        chunk_markdown(p, tmp_path)


def test_default_kb_skips_bad_note_with_warning(tmp_path, caplog):
    notes = tmp_path / "research_notes"
    notes.mkdir()
    (notes / "good.md").write_text("# G\nfine\n")
    (notes / "bad.md").write_bytes(b"# B\n\xff\xfe\xfa\n")
    with caplog.at_level("WARNING"):
        kb = build_default_kb(tmp_path, load_scenario(V1), HashEmbedder())
    assert any("bad.md" in r.getMessage() for r in caplog.records)
    assert any("good.md" in c.file for c in kb.chunks)
    assert not any("bad.md" in c.file for c in kb.chunks)


def test_search_k_zero_or_negative_is_empty():
    assert make_kb().search("dialysis", k=0) == [] and make_kb().search("dialysis", k=-1) == []


def test_url_keeps_balanced_parentheses(tmp_path):
    md = "# A\nsee https://en.wikipedia.org/wiki/Foo_(bar) and [x](https://a.org/b), also (https://c.org/d).\n"
    c = chunk_markdown(write(tmp_path, md), tmp_path)[0]
    assert c.links == ["https://en.wikipedia.org/wiki/Foo_(bar)", "https://a.org/b", "https://c.org/d"]
