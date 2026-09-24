"""Project-scope rules for the dashboard agent-switch handler.

``api_chat_slot_agent`` re-derives ``slot.project`` from the newly selected
agent's bindings, and that value becomes the next turn's subprocess cwd. Two
carve-outs keep the derivation from discarding a directory the user deliberately
chose:

* a slot filed into a project-linked sidebar folder keeps that folder's
  directory, so file search and tools stay in the folder's repo;
* an agent that resolves to the DEFAULT workspace (bound to none, or naming one
  absent from the config) leaves ``slot.project`` alone, so such a pick does not
  silently move the chat to the default workspace root.

These tests pin both carve-outs, plus the cases that must keep resetting.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_app_with_agent_routes, _make_state
from dashboard_owner_helpers import as_owner

from kiro_crew.config.loader import KiroCrewAgentConfig, KiroCrewConfig, ResolvedBindings
from kiro_crew.execution_context import ExecutionContext, MemoryStoreRef

MOD = "kiro_crew.dashboard.chat_handlers"

_WORKSPACE_DEFAULT_DIR = "/workspace/default"
_WORKSPACE_DEV_DIR = "/workspace/dev"


def _stub_agent_resolution(
    monkeypatch, *, ws_name: str, workspace_dir: str, default_workspace: str = "default"
) -> None:
    """Make the switch resolve ``dev`` as a config alias bound to *ws_name*.

    *default_workspace* is set explicitly rather than left to the MagicMock: the
    handler compares ``ws_name`` against it, and an auto-attribute is never equal
    to anything, so a test would pass whether or not that comparison exists.
    """
    mock_cfg = KiroCrewConfig()
    # A config alias, so the project-scope carve-out above does not apply.
    mock_cfg.agents = {"dev": KiroCrewAgentConfig(workspace=ws_name)}
    mock_cfg.default_workspace = default_workspace

    mock_bindings = ResolvedBindings(
        workspace_dir=Path(workspace_dir),
        effective_memory_config={},
        memory_store_name="default",
        requested_resolved=True,
        resolved_alias="dev",
        kiro_agent="kirocrew",
        selection_kind="member",
        execution_context=ExecutionContext(
            None, MemoryStoreRef("default"), "member", "kirocrew", selection_name="dev"
        ),
    )

    monkeypatch.setattr(f"{MOD}.KiroCrewConfig.load", lambda: mock_cfg)
    monkeypatch.setattr(
        f"{MOD}.resolve_agent_bindings", lambda cfg, name, project_dir=None, **kwargs: mock_bindings
    )
    monkeypatch.setattr(f"{MOD}._workspace_name_for_dir", lambda cfg, ws_dir: ws_name)
    monkeypatch.setattr(f"{MOD}.warm_project_agent_names", AsyncMock())
    monkeypatch.setattr(f"{MOD}.cached_project_agent_names", lambda project_dir: frozenset())
    monkeypatch.setattr(f"{MOD}.default_project_dir", lambda ws: workspace_dir)


class TestChatSlotAgentProjectScope:
    @pytest.mark.asyncio
    async def test_agent_switch_preserves_folder_project_dir(self, tmp_path, monkeypatch):
        """A slot in a project-linked folder keeps that folder's directory.

        Without this the agent pick retargets the slot at the new agent's
        workspace default, and the folder's repo silently stops being the cwd.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        folder_dir = tmp_path / "folder-repo"
        folder_dir.mkdir()
        state = _make_state(tmp_path)
        state._folders = [{"id": "f1", "name": "Repo", "project_dir": str(folder_dir)}]
        slot = state.get_or_create_slot("s1")
        slot.folder_id = "f1"
        slot.project = str(folder_dir)
        state.sessions.reset = AsyncMock()
        _stub_agent_resolution(monkeypatch, ws_name="dev-ws", workspace_dir=_WORKSPACE_DEV_DIR)

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "dev"})
            assert resp.status == 200
            assert slot.project == str(
                folder_dir
            ), f"agent switch clobbered the folder project: {slot.project!r}"

    @pytest.mark.asyncio
    async def test_agent_switch_inherits_ancestor_folder_project_dir(self, tmp_path, monkeypatch):
        """The folder project is inherited from the nearest configured ancestor."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        folder_dir = tmp_path / "parent-repo"
        folder_dir.mkdir()
        state = _make_state(tmp_path)
        state._folders = [
            {"id": "parent", "name": "Parent", "project_dir": str(folder_dir)},
            {"id": "child", "name": "Child", "parent_id": "parent"},
        ]
        slot = state.get_or_create_slot("s1")
        slot.folder_id = "child"
        slot.project = str(folder_dir)
        state.sessions.reset = AsyncMock()
        _stub_agent_resolution(monkeypatch, ws_name="dev-ws", workspace_dir=_WORKSPACE_DEV_DIR)

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "dev"})
            assert resp.status == 200
            assert slot.project == str(folder_dir)

    @pytest.mark.asyncio
    async def test_agent_switch_delivers_a_member_bound_folder_to_a_chat_running_as_the_member(
        self, tmp_path, monkeypatch
    ):
        """Owner-scoped delivery: a folder a crew member bound at create confers
        its directory on chats running AS the member -- spelled
        ``member:<store>`` exactly as the steering gate spells it -- from the
        chat's FIRST turn. A real member (private V2 store) is provisioned; the
        chat is opened in the member's folder AS the member and lands the
        member's directory at the mint (the principal is read off the
        selection this create records, not off the execution record, which
        does not exist yet), and its same-name reset -- an agent CHANGE on a
        member-pinned chat is refused earlier -- re-resolves the folder and
        keeps it. Before the fix the mint committed the workspace default and
        only the reset landed the member's directory, so the member's first
        turn ran in the wrong checkout."""
        from test_chat_agent_selection import TEMPLATE, _template_chat

        state, _template_slot, private_store = await _template_chat(
            tmp_path, monkeypatch, first_turn=False
        )
        folder_dir = tmp_path / "reviews-repo"
        folder_dir.mkdir()
        state._folders = [
            {
                "id": "reviews",
                "name": "Reviews",
                "owner_app": f"member:{private_store}",
                "project_dir": str(folder_dir),
            }
        ]
        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post(
                "/api/chat/slots",
                json={
                    "name": "worker",
                    "agent": TEMPLATE,
                    "agent_kind": "member",
                    "folder_id": "reviews",
                },
            )
            assert resp.status == 200, await resp.text()
            slot = state._slots["worker"]
            assert slot.folder_id == "reviews"
            assert slot.project == str(
                folder_dir
            ), f"the member's chat did not start in the member's directory: {slot.project!r}"
            resp = await client.post(
                "/api/chat/slots/worker/agent", json={"agent": TEMPLATE, "agent_kind": "member"}
            )
            assert resp.status == 200, await resp.text()
            assert slot.project == str(
                folder_dir
            ), f"the same-name reset lost the member's directory: {slot.project!r}"

    @pytest.mark.asyncio
    async def test_an_invalid_member_binding_does_not_commit_the_persons_earlier_resolution(
        self, tmp_path, monkeypatch
    ):
        """A member-bound folder whose stored directory has since become invalid
        (deleted here), nested under a person-bound folder that validates. The
        member's chat created there is re-resolved AS the member; that resolve
        refuses on the member's own row, and the refusal must leave the slot's
        project exactly as it was before the re-resolve -- empty, so the
        workspace default lands. Before the fix the error branch only warned,
        and the person-scoped value resolved earlier in the request (the
        ancestor's directory) was committed as the member chat's project: a
        directory the member's own binding was meant to override, persisted
        under a slot that then ran there."""
        from test_chat_agent_selection import TEMPLATE, _template_chat

        state, template_slot, private_store = await _template_chat(
            tmp_path, monkeypatch, first_turn=False
        )
        workspace_default = template_slot.project
        person_dir = tmp_path / "persons-repo"
        person_dir.mkdir()
        gone = tmp_path / "members-repo-since-deleted"
        state._folders = [
            {"id": "mine", "name": "Mine", "project_dir": str(person_dir)},
            {
                "id": "reviews",
                "name": "Reviews",
                "parent_id": "mine",
                "owner_app": f"member:{private_store}",
                "project_dir": str(gone),
            },
        ]
        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post(
                "/api/chat/slots",
                json={
                    "name": "worker",
                    "agent": TEMPLATE,
                    "agent_kind": "member",
                    "folder_id": "reviews",
                },
            )
            assert resp.status == 200, await resp.text()
            slot = state._slots["worker"]
            assert slot.folder_id == "reviews"
            assert slot.project != str(person_dir), (
                "an invalid member binding committed the person's earlier "
                f"resolution as the member chat's project: {slot.project!r}"
            )
            assert (
                slot.project == workspace_default
            ), f"expected the workspace default {workspace_default!r}, got {slot.project!r}"

    @pytest.mark.asyncio
    async def test_first_member_pick_on_the_persons_chat_lands_the_members_directory(
        self, tmp_path, monkeypatch
    ):
        """The switch arm of the same rule. A chat the person opened in the
        member's folder runs as the person and is withheld the member's binding
        (the mint lands the workspace default). Its FIRST pick of the member --
        allowed while the conversation is empty -- makes it run AS the member,
        so the switch resolves the folder for the member and lands the member's
        directory. Before the fix the switch read the principal off the prior
        execution record, which named nobody, skipped the member's binding and
        kept the workspace default: the member's first turn ran there, and only
        a SECOND same-name reset corrected it."""
        from test_chat_agent_selection import TEMPLATE, _template_chat

        state, _template_slot, private_store = await _template_chat(
            tmp_path, monkeypatch, first_turn=False
        )
        folder_dir = tmp_path / "reviews-repo"
        folder_dir.mkdir()
        state._folders = [
            {
                "id": "reviews",
                "name": "Reviews",
                "owner_app": f"member:{private_store}",
                "project_dir": str(folder_dir),
            }
        ]
        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post(
                "/api/chat/slots", json={"name": "mine", "folder_id": "reviews"}
            )
            assert resp.status == 200, await resp.text()
            slot = state._slots["mine"]
            assert slot.project != str(folder_dir), "the person's chat took the member's binding"
            resp = await client.post(
                "/api/chat/slots/mine/agent", json={"agent": TEMPLATE, "agent_kind": "member"}
            )
            assert resp.status == 200, await resp.text()
            assert slot.project == str(
                folder_dir
            ), f"the first member pick did not land the member's directory: {slot.project!r}"

    @pytest.mark.asyncio
    async def test_agent_switch_withholds_a_member_bound_folder_from_the_persons_slot(
        self, tmp_path, monkeypatch
    ):
        """The other half of the rule: the PERSON's slot filed in the member's
        folder (no member execution -- the person opened it) resolves nothing
        from the member's binding, so the switch lands the new agent's workspace
        default rather than the directory the member chose. Before the rule the
        member's create-time binding was handed to the person's chat here."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        folder_dir = tmp_path / "reviews-repo"
        folder_dir.mkdir()
        state = _make_state(tmp_path)
        state._folders = [
            {
                "id": "reviews",
                "name": "Reviews",
                "owner_app": "member:reviewer-store",
                "project_dir": str(folder_dir),
            }
        ]
        slot = state.get_or_create_slot("s1")
        slot.folder_id = "reviews"
        slot.project = str(folder_dir)
        state.sessions.reset = AsyncMock()
        _stub_agent_resolution(monkeypatch, ws_name="dev-ws", workspace_dir=_WORKSPACE_DEV_DIR)
        monkeypatch.setattr("kiro_crew.execution_context.read_session_execution", lambda _key: None)

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "dev"})
            assert resp.status == 200, await resp.text()
            assert (
                slot.project == _WORKSPACE_DEV_DIR
            ), f"the member's binding reached the person's slot: {slot.project!r}"

    @pytest.mark.asyncio
    async def test_agent_switch_preserves_project_when_agent_resolves_to_default(
        self, tmp_path, monkeypatch
    ):
        """An agent resolving to the DEFAULT workspace leaves the project alone.

        ``_workspace_name_for_dir`` answers the literal "default" for an agent
        bound to no workspace and for one naming a workspace absent from the
        config, so without the gate every such pick discards the user's choice.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        picked = tmp_path / "picked"
        picked.mkdir()
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.project = str(picked)
        state.sessions.reset = AsyncMock()
        _stub_agent_resolution(monkeypatch, ws_name="default", workspace_dir=_WORKSPACE_DEFAULT_DIR)

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "dev"})
            assert resp.status == 200
            assert slot.project == str(
                picked
            ), f"default-resolving agent clobbered the project: {slot.project!r}"

    @pytest.mark.asyncio
    async def test_agent_switch_preserves_project_on_a_RENAMED_default_workspace(
        self, tmp_path, monkeypatch
    ):
        """The same fallback, spelled differently.

        On an install that renames its default workspace, the resolver falls back
        to that NAME rather than the literal "default", so a literal-only gate lets
        the fallback through and the project is clobbered on exactly the installs
        that configured a default.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        picked = tmp_path / "picked"
        picked.mkdir()
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.project = str(picked)
        state.sessions.reset = AsyncMock()
        _stub_agent_resolution(
            monkeypatch,
            ws_name="house-default",
            workspace_dir=_WORKSPACE_DEFAULT_DIR,
            default_workspace="house-default",
        )

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "dev"})
            assert resp.status == 200
            assert slot.project == str(
                picked
            ), f"renamed-default agent clobbered the project: {slot.project!r}"

    @pytest.mark.asyncio
    async def test_agent_switch_retargets_project_for_named_workspace(self, tmp_path, monkeypatch):
        """A folder-less slot still follows a NAMED workspace (control).

        The reset is deliberate for this case — file search must not stay scoped
        to the previous workspace — so the two carve-outs above must not widen
        into an unconditional preserve.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.project = "/old/project"
        state.sessions.reset = AsyncMock()
        _stub_agent_resolution(monkeypatch, ws_name="dev-ws", workspace_dir=_WORKSPACE_DEV_DIR)

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "dev"})
            assert resp.status == 200
            assert slot.project == _WORKSPACE_DEV_DIR

    @pytest.mark.asyncio
    async def test_agent_switch_survives_unusable_folder_project(self, tmp_path, monkeypatch):
        """A folder whose project directory is gone must not block the switch.

        The create path answers 400 for an invalid folder project; doing that
        here would make the agent permanently unswitchable for any folder whose
        directory was moved or deleted, so the switch falls through instead.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state._folders = [
            {"id": "f1", "name": "Gone", "project_dir": str(tmp_path / "deleted-repo")}
        ]
        slot = state.get_or_create_slot("s1")
        slot.folder_id = "f1"
        slot.project = "/old/project"
        state.sessions.reset = AsyncMock()
        _stub_agent_resolution(monkeypatch, ws_name="dev-ws", workspace_dir=_WORKSPACE_DEV_DIR)

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "dev"})
            assert resp.status == 200
            assert slot.project == _WORKSPACE_DEV_DIR
