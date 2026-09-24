"""``project_dir`` on the folder tools, driven through the REAL folder routes.

``test_mcp_dashboard_folders.py`` patches the HTTP helpers and pins the tool's
call shape. This module closes the other half of the tool's claim — that the
tool surfaces the endpoint's validation rather than a copy of it — by wiring
``mcp_dashboard``'s ``_get`` / ``_post`` / ``_patch`` to a live aiohttp test
server running ``chat_folders``' own handlers over a real ``DashboardState``,
then reading the store the routes wrote. The slot-create route is included so
the inheritance the feature exists for is observed end to end: a folder the
tool bound is one a session created inside it inherits from.

The tool is synchronous (it is a stdio MCP server), so it runs on a worker
thread while the bridge hands each request back to the test's event loop with
``run_coroutine_threadsafe`` — the same request/response contract
``mcp_core._send`` presents (a 2xx body verbatim; a 4xx collapsed to
``{"error", "code"}``).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_app_with_agent_routes, _make_folder_app, _make_state

from kiro_crew.dashboard.chat_folders import _resolve_folder_project_dir
from kiro_crew.dashboard.token_auth import MEMBER_CHAT_PRINCIPAL_KEY
from kiro_crew.mcp_dashboard import _call_tool_inner

CALLER = "chat-1-100"


class _Bridge:
    """``mcp_core`` request helpers re-targeted at an aiohttp ``TestClient``.

    ``default_session_key`` mirrors the real helpers' default: ``_get`` /
    ``_post`` / ``_patch`` resolve the caller's own key when none is passed, so a
    read the tool makes without one (the slot roster its tree-shaping gate scopes
    the caller by) still reaches the routes AS the caller. ``_call`` sets it to
    the key it hands the strict resolver.
    """

    def __init__(self, client: TestClient, loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self._loop = loop
        self.default_session_key: str | None = None

    async def _request(
        self, method: str, path: str, body: dict | None, session_key: str | None
    ) -> Any:
        headers = {"X-Session-Key": session_key} if session_key else {}
        resp = await self._client.request(method, path, json=body, headers=headers)
        payload = await resp.json()
        if resp.status >= 400:
            # ``_http_error_body``'s flattening: the structured error, plus code.
            return {"error": str(payload.get("error")), "code": str(payload.get("code") or "")}
        return payload

    def _run(self, method: str, path: str, body: dict | None, session_key: str | None) -> Any:
        if session_key is None:
            session_key = self.default_session_key
        return asyncio.run_coroutine_threadsafe(
            self._request(method, path, body, session_key), self._loop
        ).result(timeout=30)

    # Signatures mirror ``mcp_core._get`` / ``_post`` / ``_patch``.
    def get(self, path: str, session_key: str | None = None, *, timeout: float = 10) -> Any:
        return self._run("GET", path, None, session_key)

    def post(
        self,
        path: str,
        body: dict | None = None,
        *,
        timeout: float = 30,
        session_key: str | None = None,
    ) -> dict:
        return self._run("POST", path, body or {}, session_key)

    def patch(self, path: str, body: dict | None = None, *, session_key: str | None = None) -> dict:
        return self._run("PATCH", path, body or {}, session_key)


async def _call(
    bridge: _Bridge, name: str, args: dict[str, Any], *, caller_key: str = f"dashboard:{CALLER}"
) -> str:
    """Run one tool call on a worker thread against the live routes.

    ``caller_key`` is the verified key the strict resolver hands the tool -- the
    dashboard slot by default, or another namespace (a ``channel:`` agent) to
    drive the routes as that principal.
    """
    bridge.default_session_key = caller_key
    with (
        patch("kiro_crew.mcp_dashboard._get", side_effect=bridge.get),
        patch("kiro_crew.mcp_dashboard._post", side_effect=bridge.post),
        patch("kiro_crew.mcp_dashboard._patch", side_effect=bridge.patch),
        patch(
            "kiro_crew.mcp_core._resolve_session_key_strict",
            return_value=caller_key,
        ),
    ):
        return await asyncio.to_thread(_call_tool_inner, name, args)


def _folder(state: Any, fid: str) -> dict[str, Any]:
    return next(f for f in state._folders if f["id"] == fid)


def _created_id(out: str) -> str:
    assert "(id=" in out, out
    return out.split("(id=", 1)[1].split(")", 1)[0]


MEMBER = "member:reviewer-store"


def _member_folder_app(state: Any, principal: str = MEMBER) -> web.Application:
    """The folder routes as an admitted crew MEMBER reaches them: the chat-route
    gate (``handlers/_shared.py``) stamps the verified ``member:<store>``
    principal on the request, and ``folder_principal`` reads that key. The
    member's own slot carries no app, so the tool's tree-shaping gate scopes it
    like the person; the routes are what tell the two apart."""
    app = _make_folder_app(state)

    @web.middleware
    async def _stamp_member(request: web.Request, handler: Any) -> Any:
        request[MEMBER_CHAT_PRINCIPAL_KEY] = principal
        return await handler(request)

    app.middlewares.append(_stamp_member)
    return app


def _pin_slot_create_defaults(monkeypatch: Any, tmp_path: Any) -> str:
    """The pins the route's own inheritance test uses (test_dashboard_chat): no
    configured default project, no eager spawn, a string default agent. Returns
    the workspace default a chat falls back to when its folder confers none."""
    mock_cfg = MagicMock()
    mock_cfg.dashboard.default_project = ""
    mock_cfg.default_agent = ""
    monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", lambda: mock_cfg)
    fallback = str(tmp_path / "workspace-default")
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_handlers.default_project_dir", lambda _workspace: fallback
    )
    monkeypatch.setattr(
        "kiro_crew.dashboard.chat_handlers.schedule_eager_spawn", lambda *_args, **_kwargs: None
    )
    return fallback


@pytest.fixture
def state(tmp_path: Any, monkeypatch: Any) -> Any:
    monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
    st = _make_state(tmp_path)
    # The caller's own live slot: the tree-shaping gate scopes the caller by
    # finding this row, and the routes refuse a ``dashboard:`` key naming a slot
    # that is gone. Created with no app, so the caller is the person.
    st.get_or_create_slot(CALLER)
    return st


class TestCreateThroughTheRealRoute:
    @pytest.mark.asyncio
    async def test_a_valid_project_dir_is_stored_and_reported_canonically(
        self, state: Any, tmp_path: Any
    ) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        link = tmp_path / "proj-link"
        link.symlink_to(proj, target_is_directory=True)
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge, "chat_folder_create", {"name": "Proj", "project_dir": str(link)}
            )
        assert out.startswith("Created folder `Proj`"), out
        stored = _folder(state, _created_id(out))
        # The endpoint's validator canonicalised the path; the store and the
        # tool's report agree on the resolved form, which is what a session
        # created here will inherit.
        assert stored["project_dir"] == os.path.realpath(str(proj))
        assert f"Project directory: {os.path.realpath(str(proj))}" in out

    @pytest.mark.asyncio
    async def test_a_session_created_in_the_folder_inherits_the_binding(
        self, state: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """The point of the feature, observed: create the folder with the tool,
        then open a chat in it the way the dashboard does, and the slot starts
        with the folder's project."""
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge, "chat_folder_create", {"name": "Proj", "project_dir": str(proj)}
            )
        fid = _created_id(out)

        _pin_slot_create_defaults(monkeypatch, tmp_path)
        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots", json={"name": "in-proj", "folder_id": fid})
            data = await resp.json()
        assert resp.status == 200, data
        assert data["folder_id"] == fid
        assert data["project"] == os.path.realpath(str(proj))
        assert state._slots["in-proj"].project == os.path.realpath(str(proj))

    @pytest.mark.asyncio
    async def test_a_crew_member_binds_at_create_and_only_its_own_chats_inherit(
        self, state: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """Pin for the principal that KEEPS the create-time binding: a crew
        member on a private store -- a conductor standing up a project folder
        for its workers -- binds the folder it creates and the folder is
        stamped as its own. Delivery is owner-scoped, the steering gate's rule:
        the member's own chats resolve the binding; the PERSON's chat opened in
        that folder from the sidebar inherits nothing from it and starts in the
        workspace default, so a member's create cannot pick the directory the
        person's session is launched in."""
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_member_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge, "chat_folder_create", {"name": "Reviews", "project_dir": str(proj)}
            )
        assert out.startswith("Created folder `Reviews`"), out
        fid = _created_id(out)
        assert _folder(state, fid)["owner_app"] == MEMBER
        assert _folder(state, fid)["project_dir"] == os.path.realpath(str(proj))

        fallback = _pin_slot_create_defaults(monkeypatch, tmp_path)
        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post(
                "/api/chat/slots", json={"name": "in-reviews", "folder_id": fid}
            )
            data = await resp.json()
        assert resp.status == 200, data
        assert data["folder_id"] == fid
        assert data["project"] == fallback
        assert state._slots["in-reviews"].project == fallback
        # The member's own chat there resolves the binding.
        assert _resolve_folder_project_dir(state._folders, fid, slot_app=MEMBER) == (
            os.path.realpath(str(proj)),
            None,
        )

    @pytest.mark.asyncio
    async def test_a_sensitive_path_is_refused_with_the_routes_text_and_no_folder(
        self, state: Any
    ) -> None:
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge, "chat_folder_create", {"name": "Keys", "project_dir": "~/.ssh"}
            )
        assert out == "Error: project_dir refers to a sensitive path"
        assert state._folders == []

    @pytest.mark.asyncio
    async def test_a_missing_directory_is_refused_with_the_routes_text_and_no_folder(
        self, state: Any, tmp_path: Any
    ) -> None:
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge,
                "chat_folder_create",
                {"name": "Ghost", "project_dir": str(tmp_path / "does-not-exist")},
            )
        assert out == "Error: Project directory must be an existing directory"
        assert state._folders == []

    @pytest.mark.asyncio
    async def test_a_relative_path_is_refused_with_the_routes_text_and_no_folder(
        self, state: Any
    ) -> None:
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge, "chat_folder_create", {"name": "Rel", "project_dir": "projects/rel"}
            )
        assert out == "Error: Project directory must be an absolute path"
        assert state._folders == []

    @pytest.mark.asyncio
    async def test_a_create_without_project_dir_stores_an_empty_binding(self, state: Any) -> None:
        """Pin: a call without ``project_dir`` is byte-for-byte the same request, so the
        store row it produces is the one it always produced."""
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(bridge, "chat_folder_create", {"name": "Plain"})
        stored = _folder(state, _created_id(out))
        assert stored["project_dir"] == ""
        assert out == f"Created folder `Plain` (id={stored['id']})."


class TestUpdateThroughTheRealRoute:
    @pytest.mark.asyncio
    async def test_sets_then_clears(self, state: Any, tmp_path: Any) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(bridge, "chat_folder_create", {"name": "Work"})
            fid = _created_id(made)
            assert _folder(state, fid)["project_dir"] == ""

            out = await _call(
                bridge, "chat_folder_update", {"folder": "Work", "project_dir": str(proj)}
            )
            assert out.startswith("Set the project directory of `Work`"), out
            assert _folder(state, fid)["project_dir"] == os.path.realpath(str(proj))

            out = await _call(bridge, "chat_folder_update", {"folder": fid, "project_dir": ""})
            assert out.startswith("Cleared the project directory of `Work`"), out
            assert _folder(state, fid)["project_dir"] == ""

    @pytest.mark.asyncio
    async def test_a_null_project_dir_clears_through_the_real_route(
        self, state: Any, tmp_path: Any
    ) -> None:
        """``null`` reaches the endpoint as the ``""`` clear, never as the string
        ``"None"`` -- which the route would refuse as a relative path, leaving
        the old binding in place behind an error."""
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(
                bridge, "chat_folder_create", {"name": "Work", "project_dir": str(proj)}
            )
            fid = _created_id(made)
            assert _folder(state, fid)["project_dir"] == os.path.realpath(str(proj))

            out = await _call(bridge, "chat_folder_update", {"folder": fid, "project_dir": None})
            assert out.startswith("Cleared the project directory of `Work`"), out
            assert _folder(state, fid)["project_dir"] == ""

    @pytest.mark.asyncio
    async def test_an_invalid_path_changes_nothing(self, state: Any, tmp_path: Any) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(
                bridge, "chat_folder_create", {"name": "Work", "project_dir": str(proj)}
            )
            fid = _created_id(made)
            for bad, text in (
                ("~/.aws", "project_dir refers to a sensitive path"),
                (str(tmp_path / "gone"), "Project directory must be an existing directory"),
            ):
                out = await _call(
                    bridge, "chat_folder_update", {"folder": "Work", "project_dir": bad}
                )
                assert out == f"Error: {text}"
                assert _folder(state, fid)["project_dir"] == os.path.realpath(str(proj))

    @pytest.mark.asyncio
    async def test_an_app_cannot_bind_the_persons_folder(self, state: Any, tmp_path: Any) -> None:
        """Ownership is the endpoint's, under the store lock: the app's PATCH
        reaches it under the app's verified key and comes back refused. The tool
        adds no create-time hint for an app: that path is closed to it too."""
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(bridge, "chat_folder_create", {"name": "Mine"})
            fid = _created_id(made)
            assert _folder(state, fid).get("owner_app", "") == ""

            # The same caller slot, now owned by an app (what an app-created
            # session's row carries), so the endpoint derives that app.
            state._slots[CALLER]._app = "issue-radar"
            out = await _call(
                bridge, "chat_folder_update", {"folder": "Mine", "project_dir": str(proj)}
            )
        assert out == (
            "Error: an app cannot set or clear a folder's project directory - ask the person"
        ), out
        assert "chat_folder_create with project_dir" not in out
        assert _folder(state, fid)["project_dir"] == ""

    @pytest.mark.asyncio
    async def test_an_app_cannot_bind_at_create_and_a_binding_on_its_folder_is_inert(
        self, state: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """An app has no binding verb at all: ``project_dir`` on its create is
        refused through the real route with nothing created, and a binding the
        PERSON later puts on the app's folder confers nothing on a chat opened
        there -- the slot starts on the workspace default, not in the directory
        the folder names. So no route to an app-owned folder makes the gateway
        launch a session in an app-chosen directory."""
        proj = tmp_path / "proj"
        proj.mkdir()
        state._slots[CALLER]._app = "issue-radar"
        # The person's own live slot, for the person's PATCH below.
        state.get_or_create_slot("chat-2-200")
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge, "chat_folder_create", {"name": "Radar output", "project_dir": str(proj)}
            )
            assert out == (
                "Error: an app cannot set or clear a folder's project directory - ask the person"
            ), out
            assert state._folders == []

            made = await _call(bridge, "chat_folder_create", {"name": "Radar output"})
            fid = _created_id(made)
            assert _folder(state, fid)["owner_app"] == "issue-radar"
            assert _folder(state, fid)["project_dir"] == ""
            bound = await client.patch(
                f"/api/chat/folders/{fid}",
                json={"project_dir": str(proj)},
                headers={"X-Session-Key": "dashboard:chat-2-200"},
            )
            assert bound.status == 200, await bound.text()
            assert _folder(state, fid)["project_dir"] == os.path.realpath(str(proj))

        fallback = _pin_slot_create_defaults(monkeypatch, tmp_path)
        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots", json={"name": "in-radar", "folder_id": fid})
            data = await resp.json()
        assert resp.status == 200, data
        assert data["folder_id"] == fid
        assert data["project"] == fallback
        assert state._slots["in-radar"].project == fallback
        assert _resolve_folder_project_dir(state._folders, fid) == ("", None)

    @pytest.mark.asyncio
    async def test_an_app_cannot_rebind_its_folder_holding_the_persons_chat(
        self, state: Any, tmp_path: Any
    ) -> None:
        """The person filed one of their own chats into the app's folder. That
        slot picks the folder's binding up on its next agent switch, so the
        app's rebind would decide the person's project -- refused through the
        real route, binding unchanged."""
        proj = tmp_path / "proj"
        proj.mkdir()
        state._slots[CALLER]._app = "issue-radar"
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(bridge, "chat_folder_create", {"name": "Radar output"})
            fid = _created_id(made)
            theirs = state.get_or_create_slot("chat-2-200")
            theirs.folder_id = fid
            out = await _call(
                bridge,
                "chat_folder_update",
                {"folder": "Radar output", "project_dir": str(proj)},
            )
        assert out == (
            "Error: an app cannot set or clear a folder's project directory - ask the person"
        ), out
        assert _folder(state, fid)["project_dir"] == ""

    @pytest.mark.asyncio
    async def test_an_app_cannot_bind_its_folder_once_the_person_nested_one_inside(
        self, state: Any, tmp_path: Any
    ) -> None:
        """One instance of the same rule: a chat the person opens in the folder
        they nested inside the app's would inherit the app's binding."""
        proj = tmp_path / "proj"
        proj.mkdir()
        state._slots[CALLER]._app = "issue-radar"
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(bridge, "chat_folder_create", {"name": "Radar output"})
            fid = _created_id(made)
            assert _folder(state, fid)["owner_app"] == "issue-radar"
            # The person nests a folder of their own inside the app's.
            state._folders.append(
                {"id": "fldr0000theirs", "name": "Theirs", "parent_id": fid, "owner_app": ""}
            )
            out = await _call(
                bridge,
                "chat_folder_update",
                {"folder": "Radar output", "project_dir": str(proj)},
            )
        assert out == (
            "Error: an app cannot set or clear a folder's project directory - ask the person"
        ), out
        assert _folder(state, fid)["project_dir"] == ""

    @pytest.mark.asyncio
    async def test_a_crew_member_binds_at_create_but_cannot_change_its_own_folder_afterwards(
        self, state: Any, tmp_path: Any
    ) -> None:
        """The capability a crew member keeps is the CREATE-time binding; an
        existing folder's binding -- even its own, holding nothing but its own
        work -- is the person's to change, and the refusal (with the tool's
        hint) names the path that IS open to it."""
        proj = tmp_path / "proj"
        proj.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        async with TestClient(TestServer(_member_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(
                bridge, "chat_folder_create", {"name": "Reviews", "project_dir": str(proj)}
            )
            fid = _created_id(made)
            assert _folder(state, fid)["owner_app"] == MEMBER
            assert _folder(state, fid)["project_dir"] == os.path.realpath(str(proj))
            for value in (str(other), ""):
                out = await _call(
                    bridge, "chat_folder_update", {"folder": "Reviews", "project_dir": value}
                )
                assert out.startswith(
                    "Error: a crew member cannot change an existing folder's project directory"
                ), out
                assert "chat_folder_create with project_dir" in out
                assert _folder(state, fid)["project_dir"] == os.path.realpath(str(proj))


class TestAChannelAgentThroughTheRealRoute:
    """A Channels agent's key (``channel:<channel_id>:<agent_id>``) names no slot
    and no app, so the tree-shaping gate scopes it as the person and the routes
    derive no principal for it. The binding fence is the endpoint's, on both
    paths, keyed on the channel key itself -- observed here through the tool."""

    CHANNEL = "channel:chan-000001:helper"

    @pytest.mark.asyncio
    async def test_a_channel_agent_cannot_bind_at_create_or_update(
        self, state: Any, tmp_path: Any
    ) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge,
                "chat_folder_create",
                {"name": "Bound", "project_dir": str(proj)},
                caller_key=self.CHANNEL,
            )
            assert out == (
                "Error: a channel agent cannot set or clear a folder's project directory - "
                "ask the person"
            ), out
            assert not any(f["name"] == "Bound" for f in state._folders)

            # The person creates an unbound folder; the channel agent may not
            # bind it, and is NOT pointed at a create-time binding it is refused
            # just the same.
            made = await _call(bridge, "chat_folder_create", {"name": "Work"})
            fid = _created_id(made)
            out = await _call(
                bridge,
                "chat_folder_update",
                {"folder": "Work", "project_dir": str(proj)},
                caller_key=self.CHANNEL,
            )
            assert out == (
                "Error: a channel agent cannot set or clear a folder's project directory - "
                "ask the person"
            ), out
            assert "chat_folder_create" not in out
            assert _folder(state, fid)["project_dir"] == ""

    @pytest.mark.asyncio
    async def test_a_channel_agents_unbound_create_still_lands(
        self, state: Any, tmp_path: Any
    ) -> None:
        """Scope pin: the fence is on the binding, not on the channel agent's
        other folder writes."""
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge, "chat_folder_create", {"name": "Notes"}, caller_key=self.CHANNEL
            )
        assert out.startswith("Created folder `Notes`"), out

    @pytest.mark.asyncio
    async def test_a_channel_agent_cannot_move_the_persons_folder_under_a_binding(
        self, state: Any, tmp_path: Any
    ) -> None:
        """The third path: ``chat_folder_move`` is not a blocked channel-agent
        tool, and a channel key reaches the person's folders as the person. The
        person binds one folder and files a chat in another, unbound one; the
        channel agent moves the unbound folder under the bound one. Before: the
        move lands and the chat resolves the bound directory on its next agent
        switch. After: refused with the binding-fence text, nothing inherited."""
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            bound = _created_id(
                await _call(
                    bridge, "chat_folder_create", {"name": "Bound", "project_dir": str(proj)}
                )
            )
            work = _created_id(await _call(bridge, "chat_folder_create", {"name": "Work"}))
            theirs = state.get_or_create_slot("chat-2-200")
            theirs.folder_id = work

            out = await _call(
                bridge,
                "chat_folder_move",
                {"folder": "Work", "new_parent": "Bound"},
                caller_key=self.CHANNEL,
            )
        assert out.startswith(
            "Error: a crew member or channel agent cannot move a folder where its "
            "sessions would inherit a different project directory"
        ), out
        assert _folder(state, work)["parent_id"] == ""
        assert _resolve_folder_project_dir(state._folders, work) == ("", None)
        assert _folder(state, bound)["project_dir"] == os.path.realpath(str(proj))

    @pytest.mark.asyncio
    async def test_a_slack_driven_session_is_fenced_the_same_way(
        self, state: Any, tmp_path: Any
    ) -> None:
        """A turn driven from a messaging-transport thread presents that key
        (``slack:<ts>``): no slot, no app, the same words from a thread other
        people are in -- fenced on all three paths through the tools, in the
        channel agent's words, and not pointed at a create it is refused too."""
        slack = "slack:1785370133.085469"
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge,
                "chat_folder_create",
                {"name": "Bound", "project_dir": str(proj)},
                caller_key=slack,
            )
            assert out.startswith("Error: a channel agent cannot set or clear"), out
            assert not any(f["name"] == "Bound" for f in state._folders)
            bound = _created_id(
                await _call(
                    bridge, "chat_folder_create", {"name": "Bound", "project_dir": str(proj)}
                )
            )
            work = _created_id(await _call(bridge, "chat_folder_create", {"name": "Work"}))
            out = await _call(
                bridge,
                "chat_folder_update",
                {"folder": "Work", "project_dir": str(proj)},
                caller_key=slack,
            )
            assert out.startswith("Error: a channel agent cannot set or clear"), out
            assert "chat_folder_create" not in out
            assert _folder(state, work)["project_dir"] == ""
            out = await _call(
                bridge,
                "chat_folder_move",
                {"folder": "Work", "new_parent": "Bound"},
                caller_key=slack,
            )
            assert out.startswith("Error: a crew member or channel agent cannot move a folder"), out
            assert _folder(state, work)["parent_id"] == ""
            # Not this rule's concern: an unbound create by the same key lands.
            made = await _call(bridge, "chat_folder_create", {"name": "Notes"}, caller_key=slack)
            assert made.startswith("Created folder `Notes`"), made
        assert _folder(state, bound)["project_dir"] == os.path.realpath(str(proj))


class TestAMoveCannotRouteABindingOntoThePersonsChat:
    """The composition the create-only rule leaves open, through the real tools:
    a crew member binds a NEW folder at create (allowed), then moves its
    EXISTING folder -- holding one of the person's chats and its own -- under
    it. The member's own chats there would resolve the new folder's directory
    on their next agent switch (delivery is owner-scoped, so the person's chat
    never resolves a member's binding); the move rule compares what every
    principal resolves at the two places and refuses the change. (An app has
    no create-time binding to compose with, and its folders confer none.)"""

    @pytest.mark.asyncio
    async def test_the_move_is_refused_and_nothing_is_inherited(
        self, state: Any, tmp_path: Any
    ) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_member_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            bound = _created_id(
                await _call(
                    bridge, "chat_folder_create", {"name": "Bound", "project_dir": str(proj)}
                )
            )
            radar = _created_id(await _call(bridge, "chat_folder_create", {"name": "Radar output"}))
            theirs = state.get_or_create_slot("chat-2-200")
            theirs.folder_id = radar

            out = await _call(
                bridge, "chat_folder_move", {"folder": "Radar output", "new_parent": "Bound"}
            )
        assert out.startswith(
            "Error: a crew member or channel agent cannot move a folder where its "
            "sessions would inherit a different project directory"
        ), out
        assert _folder(state, radar)["parent_id"] == ""
        assert _resolve_folder_project_dir(state._folders, radar) == ("", None)
        assert _folder(state, bound)["project_dir"] == os.path.realpath(str(proj))

    @pytest.mark.asyncio
    async def test_a_folder_bound_at_create_still_moves_under_another(
        self, state: Any, tmp_path: Any
    ) -> None:
        """The member keeps organising its own bound folders: between two of its
        own bound folders nothing changes for any chat -- the member's own chats
        resolve the moved folder's binding wherever it sits, and neither place
        confers anything on the person's -- so the move lands."""
        proj = tmp_path / "proj"
        proj.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        async with TestClient(TestServer(_member_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            bound = _created_id(
                await _call(
                    bridge, "chat_folder_create", {"name": "Bound", "project_dir": str(proj)}
                )
            )
            runs = _created_id(
                await _call(
                    bridge, "chat_folder_create", {"name": "Runs", "project_dir": str(other)}
                )
            )
            out = await _call(bridge, "chat_folder_move", {"folder": "Runs", "new_parent": "Bound"})
        assert out.startswith("Moved folder"), out
        assert _folder(state, runs)["parent_id"] == bound
        assert _resolve_folder_project_dir(state._folders, runs, slot_app=MEMBER) == (
            os.path.realpath(str(other)),
            None,
        )
        # Owner-scoped: the person's chat filed there resolves nothing from it.
        assert _resolve_folder_project_dir(state._folders, runs) == ("", None)


#: The three UNC spellings Windows honours: two backslashes, two forward slashes,
#: and the extended-length ``\\?\UNC\`` form. Every one names a HOST.
_UNC_SPELLINGS = (r"\\evil\share\proj", "//evil/share/proj", r"\\?\UNC\evil\share\proj")
_UNC_REFUSAL = "Project directory must not be a network (UNC) path"


class TestAUncProjectDirIsRefusedBeforeAnyFilesystemCall:
    """``project_dir`` is agent-authored path text that reaches the gateway's
    filesystem. On a Windows gateway ``realpath``/``isdir`` on ``\\\\host\\share``
    opens an SMB connection to that host -- an outbound credential probe the
    text's author controls, with no recovery -- so a UNC-shaped value is refused
    lexically, before the first filesystem call, through the repo's one UNC
    helper (``is_unc_shape`` / ``unc_probe_allowed``, exactly as the steering
    validator in the same module and the attachment readers do). The shape is
    refused on EVERY host: path text is untrusted everywhere, and the platform is
    never consulted. Same 400 shape as the validator's other refusals; audited
    like the sensitive-path refusal.

    The refusal is an ADMISSION rule: it runs where a request names a directory
    (``_admit_project_dir`` -- the create and update routes, the scaffold's scan
    root, and the MCP tool's pre-check), not in ``_validate_project_dir``, which
    the slot-create and agent-switch read paths also run over STORED values;
    see ``TestTheReadPathResolvesAStoredValueAsBefore``.
    """

    @pytest.mark.parametrize("unc", _UNC_SPELLINGS)
    @pytest.mark.parametrize("platform", ["linux", "win32"])
    def test_the_admission_validator_refuses_every_spelling_without_touching_the_filesystem(
        self, monkeypatch: Any, unc: str, platform: str
    ) -> None:
        import sys

        import kiro_crew.dashboard.chat_folders as cf

        touched = MagicMock(side_effect=AssertionError("filesystem touched for a UNC project_dir"))
        sel_fn = MagicMock()
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(os.path, "realpath", touched)
        monkeypatch.setattr(os.path, "isdir", touched)
        monkeypatch.setattr(cf, "is_sensitive_path", touched)
        monkeypatch.setattr(cf, "unc_probe_allowed", lambda raw: False)
        monkeypatch.setattr(cf, "sel", sel_fn)
        assert cf._admit_project_dir(unc) == ("", _UNC_REFUSAL)
        touched.assert_not_called()
        kwargs = sel_fn.return_value.log_api_access.call_args.kwargs
        assert kwargs["operation"] == "chat.folder_project_dir"
        assert kwargs["outcome"] == "denied"
        assert "UNC" in kwargs["error"]

    def test_the_helper_decides_not_a_second_rule(self, monkeypatch: Any) -> None:
        """When ``unc_probe_allowed`` vouches for the share (the gateway's own data
        home on a roaming profile), the UNC refusal does not fire and the ordinary
        checks run -- the filesystem is faked so no host is ever contacted."""
        import kiro_crew.dashboard.chat_folders as cf

        monkeypatch.setattr(cf, "unc_probe_allowed", lambda raw: True)
        monkeypatch.setattr(os.path, "realpath", lambda p: p)
        monkeypatch.setattr(os.path, "isdir", lambda p: False)
        monkeypatch.setattr(cf, "is_sensitive_path", lambda p: False)
        assert cf._admit_project_dir("//evil/share/proj") == (
            "",
            "Project directory must be an existing directory",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("unc", _UNC_SPELLINGS)
    async def test_create_and_update_refuse_it_at_the_route_in_the_validators_shape(
        self, state: Any, tmp_path: Any, unc: str
    ) -> None:
        """Straight at the routes (the tool has its own pre-check, so going through
        it would prove the wrong fence): create stores nothing, update leaves the
        binding as it was; a plain absolute path and the ``""`` clear still land."""
        headers = {"X-Session-Key": f"dashboard:{CALLER}"}
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            created = await client.post(
                "/api/chat/folders", json={"name": "Share", "project_dir": unc}, headers=headers
            )
            assert created.status == 400, await created.text()
            assert await created.json() == {"error": _UNC_REFUSAL}
            assert state._folders == []
            plain = await client.post(
                "/api/chat/folders",
                json={"name": "Plain", "project_dir": str(tmp_path)},
                headers=headers,
            )
            assert plain.status == 201, await plain.text()
            fid = (await plain.json())["id"]
            updated = await client.patch(
                f"/api/chat/folders/{fid}", json={"project_dir": unc}, headers=headers
            )
            assert updated.status == 400, await updated.text()
            assert await updated.json() == {"error": _UNC_REFUSAL}
            assert _folder(state, fid)["project_dir"] == os.path.realpath(str(tmp_path))
            cleared = await client.patch(
                f"/api/chat/folders/{fid}", json={"project_dir": ""}, headers=headers
            )
            assert cleared.status == 200, await cleared.text()
        assert _folder(state, fid)["project_dir"] == ""


def _fake_fs_for(monkeypatch: Any, path: str) -> None:
    """Make *path* look like an existing, resolvable directory to the validator
    WITHOUT the process touching it: ``realpath`` is the identity and ``isdir``
    true for that one string, and the real functions otherwise. On a Windows
    shard the real ``realpath`` on a UNC string would contact the host."""
    import kiro_crew.dashboard.chat_folders as cf

    real_realpath, real_isdir = os.path.realpath, os.path.isdir
    monkeypatch.setattr(
        os.path, "realpath", lambda p, **kw: p if p == path else real_realpath(p, **kw)
    )
    monkeypatch.setattr(os.path, "isdir", lambda p: True if p == path else real_isdir(p))
    monkeypatch.setattr(cf, "unc_probe_allowed", lambda raw: False)


class TestTheReadPathResolvesAStoredValueAsBefore:
    """``_validate_project_dir`` is also the STORED-value reader: the slot-create
    and agent-switch paths run it through ``_resolve_folder_project_dir`` over
    what ``folders.json`` holds. A folder the person bound to a network share
    before the UNC rule existed is therefore not a request to admit but a
    binding to honour: refusing it at read time would fail every ``POST
    /api/chat/slots`` in that folder with 400 and drop the agent to the
    workspace default -- a retroactive refusal with no migration. So the UNC
    refusal is admission-only, and the read path resolves a stored value
    exactly as it did before the rule; a NEW one is still refused where it is
    named (the class above).
    """

    STORED = "//legacy/share/proj"

    def test_a_stored_unc_value_still_resolves(self, monkeypatch: Any) -> None:
        import kiro_crew.dashboard.chat_folders as cf

        sel_fn = MagicMock()
        monkeypatch.setattr(cf, "sel", sel_fn)
        _fake_fs_for(monkeypatch, self.STORED)
        folders = [
            {"id": "fldr00000041", "name": "Legacy", "parent_id": "", "project_dir": self.STORED}
        ]
        assert _resolve_folder_project_dir(folders, "fldr00000041") == (self.STORED, None)
        # No refusal was audited: the read path never ran the admission rule.
        sel_fn.return_value.log_api_access.assert_not_called()

    @pytest.mark.parametrize("unc", _UNC_SPELLINGS)
    def test_no_stored_spelling_meets_the_admission_refusal(
        self, monkeypatch: Any, unc: str
    ) -> None:
        """Whatever the ordinary validation says about a stored spelling on this
        host (a backslash form is not "absolute" on POSIX), it is never the
        lexical UNC refusal -- the filesystem is faked, so no host is contacted."""
        import kiro_crew.dashboard.chat_folders as cf

        monkeypatch.setattr(cf, "unc_probe_allowed", lambda raw: False)
        monkeypatch.setattr(os.path, "realpath", lambda p, **kw: p)
        monkeypatch.setattr(os.path, "isdir", lambda p: True)
        folders = [{"id": "fldr00000041", "name": "Legacy", "parent_id": "", "project_dir": unc}]
        _resolved, err = _resolve_folder_project_dir(folders, "fldr00000041")
        assert err != _UNC_REFUSAL

    @pytest.mark.asyncio
    async def test_a_chat_still_opens_in_a_folder_bound_before_the_rule(
        self, state: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """The route the retroactive refusal would have broken: a folder whose
        STORED binding is UNC-shaped (written before the rule; admission refuses
        the shape today, so it is placed in the store directly), and a chat
        opened in it the way the dashboard does."""
        _fake_fs_for(monkeypatch, self.STORED)
        state._folders.append(
            {"id": "fldr00000041", "name": "Legacy", "parent_id": "", "project_dir": self.STORED}
        )
        mock_cfg = MagicMock()
        mock_cfg.dashboard.default_project = ""
        mock_cfg.default_agent = ""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", lambda: mock_cfg
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.default_project_dir",
            lambda _workspace: str(tmp_path / "workspace-default"),
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.schedule_eager_spawn",
            lambda *_args, **_kwargs: None,
        )
        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post(
                "/api/chat/slots", json={"name": "legacy-chat", "folder_id": "fldr00000041"}
            )
            data = await resp.json()
        assert resp.status == 200, data
        assert data["project"] == self.STORED
        assert state._slots["legacy-chat"].project == self.STORED
