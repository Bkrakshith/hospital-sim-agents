"""The Studio and scenario flows: a rewritten prompt gets its own flow before the run starts, and the user can see it."""
import pytest

from test_studio_api import H, client, env, mk  # noqa: F401  (fixtures)


async def test_a_project_uses_the_preset_flow_until_its_prompt_changes(client, monkeypatch):
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    p = await mk(client)
    r = await (await client.get(f"/api/projects/{p['id']}/flow")).json()
    assert r["source"] == "preset" and r["flow"]["id"] == "ae_waiting" and r["tree"]["kind"] == "sequence"
    assert r["tree"]["children"][0]["state"] == "pending"
    await client.put(f"/api/projects/{p['id']}", json={"scenario_prompt": "A quiet night in A&E with one anxious parent."},
                     headers=H)
    r = await (await client.get(f"/api/projects/{p['id']}/flow")).json()
    assert r["source"] == "needs_adapting"
    r = await (await client.post(f"/api/projects/{p['id']}/flow", json={}, headers=H)).json()
    assert r["source"] == "adapted" and "kept" in r["notes"]
    # the same prompt again: no new adaptation is needed
    r = await (await client.get(f"/api/projects/{p['id']}/flow")).json()
    assert r["source"] == "adapted"


async def test_starting_a_run_with_a_changed_prompt_adapts_the_flow_first(client, monkeypatch):
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    p = await mk(client)
    await client.put(f"/api/projects/{p['id']}", json={"scenario_prompt": "Busy evening, the robot is new."}, headers=H)
    r = await client.post("/api/run/start", json={"project_id": p["id"]}, headers=H)
    assert r.status == 202
    flow_file = client.env["data"] / "studio" / "projects" / p["id"] / "flow.yaml"
    assert flow_file.exists() and "prompt_sha" in flow_file.read_text()
    staged = client.env["calls"][-1]["staged"]
    assert "flow: ../flows/ae_waiting.yaml" in staged.read_text()
    await client.post("/api/run/stop", headers=H)


async def test_a_prompt_that_cannot_get_a_flow_stops_the_run_with_a_reason(client, monkeypatch):
    from wardzero_agents.flows import spec
    from wardzero_agents.studio import flows

    async def broken(*a, **k):
        raise spec.FlowError("the scenario flow could not be adapted to the prompt: bad")
    monkeypatch.setattr(flows, "adapt_flow", broken)
    p = await mk(client)
    await client.put(f"/api/projects/{p['id']}", json={"scenario_prompt": "Something else entirely."}, headers=H)
    r = await client.post("/api/run/start", json={"project_id": p["id"]}, headers=H)
    assert r.status == 400 and "Restore preset" in (await r.json())["error"]
    assert not client.env["calls"]


async def test_restoring_the_preset_prompt_goes_back_to_the_preset_flow(client, monkeypatch):
    p = await mk(client)
    preset_prompt = (await (await client.get(f"/api/projects/{p['id']}/flow")).json())["preset_prompt"]
    await client.put(f"/api/projects/{p['id']}", json={"scenario_prompt": "  " + preset_prompt + " "}, headers=H)
    r = await (await client.get(f"/api/projects/{p['id']}/flow")).json()
    assert r["source"] == "preset"
