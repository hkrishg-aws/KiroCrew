"""Ownership on the chat-folder tree-shaping endpoints.

A folder created by an app carries it in ``owner_app``; an absent key reads as
the person's, which is what makes this a field addition rather than a migration.
An app may create at the top level or inside a folder it owns, and may rename,
reparent or delete only what it owns. The person is never confined.

The scope is derived from the authenticated calling session, never the body: the
managed MCP set authenticates with the internal secret, which carries no app
claim, so an app agent's tool call arrives with ``request["app"]`` empty.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.constants import CHANNEL_SESSION_NAMESPACES
from kiro_crew.dashboard.chat_folders import (
    _admit_project_dir,
    _inherited_project_dir,
    _inherited_steering_dirs,
    _resolve_folder_project_dir,
    api_chat_folder_create,
    api_chat_folder_delete,
    api_chat_folder_update,
)
from kiro_crew.dashboard.state import DashboardState, _ChatSlot
from kiro_crew.dashboard.token_auth import MEMBER_CHAT_PRINCIPAL_KEY

# fldr…01 belongs to the person, …02 to issue-radar, …03 to another app, and
# …04 predates the field entirely (no key at all) — the legacy row.
PERSON = "fldr00000001"
RADAR = "fldr00000002"
OTHER = "fldr00000003"
LEGACY = "fldr00000004"


def _folders() -> list[dict[str, Any]]:
    return [
        {"id": PERSON, "name": "Work", "parent_id": "", "owner_app": ""},
        {"id": RADAR, "name": "Radar output", "parent_id": "", "owner_app": "issue-radar"},
        {"id": OTHER, "name": "Specs", "parent_id": "", "owner_app": "spec-builder"},
        {"id": LEGACY, "name": "Old", "parent_id": ""},
    ]


def _app_slot(key: str, app: str) -> _ChatSlot:
    slot = _ChatSlot(key)
    slot._app = app
    return slot


def _state(*slots: _ChatSlot, folders: list[dict[str, Any]] | None = None) -> DashboardState:
    state = MagicMock(spec=DashboardState)
    state._folders = _folders() if folders is None else folders
    state._slots = {s.key: s for s in slots}
    state.push_slots_update = MagicMock()
    # No archive by default: _folder_history_counts returns {} early on a falsy
    # conversation_log, which is what an app's delete consults for emptiness. A
    # bare MagicMock here would be iterated instead and raise.
    state.conversation_log = None

    async def _mutate(fn: Any, on_committed: Any = None) -> Any:
        # The real store runs the callback under a lock and hands back its
        # second element; the ownership decisions live inside that callback, so a
        # mock that never calls it would prove nothing.
        changed, value = fn(state._folders)
        if changed and on_committed is not None:
            on_committed()
        return value

    state.mutate_folders = AsyncMock(side_effect=_mutate)
    return state


def _make_app(state: DashboardState, *, member_principal: str = "") -> web.Application:
    app = web.Application()
    app["state"] = state

    @web.middleware
    async def _publish_app(request: web.Request, handler: Any) -> Any:
        # Stands in for the token middleware. Empty for the internal-secret
        # (MCP) transport, which is the path an app agent's tool call takes.
        request["app"] = ""
        if member_principal:
            # What the chat-route gate stamps on the VERIFIED scope when it
            # admits a crew member (``handlers/_shared.py``); the handler's
            # ``folder_principal`` reads exactly this key.
            request[MEMBER_CHAT_PRINCIPAL_KEY] = member_principal
        return await handler(request)

    app.middlewares.append(_publish_app)
    app.router.add_post("/api/chat/folders", api_chat_folder_create)
    app.router.add_patch("/api/chat/folders/{id}", api_chat_folder_update)
    app.router.add_delete("/api/chat/folders/{id}", api_chat_folder_delete)
    return app


def _by_id(state: DashboardState, fid: str) -> dict[str, Any] | None:
    return next((f for f in state._folders if f["id"] == fid), None)


class TestOrderIsStoredVerbatim:
    """The endpoint stores whatever int the body carries, sign included.

    ``chat_folder_move``'s free-slot placement puts a folder ahead of the first
    sibling by writing ``first.order - 1``, which is NEGATIVE once the sidebar has
    renumbered a set from 0 — the ordinary case. Nothing in the tool layer can make
    that work if the endpoint clamps or rejects it, and the tool writes it as the
    single request that keeps a reposition from landing half-applied.
    """

    @pytest.mark.asyncio
    async def test_a_negative_order_is_accepted(self) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"order": -1},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, PERSON)["order"] == -1

    @pytest.mark.asyncio
    async def test_a_gap_midpoint_is_accepted(self) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"order": 5},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, PERSON)["order"] == 5

    @pytest.mark.asyncio
    async def test_a_duplicate_order_is_not_refused(self) -> None:
        """Two siblings may share a number; the name tie-break resolves them.

        The free-slot check treats equal neighbours as no room precisely because
        the store allows this, so the allowance has to be pinned.
        """
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            first = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"order": 7},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            second = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"order": 7},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert (first.status, second.status) == (200, 200)
        assert _by_id(state, PERSON)["order"] == 7
        assert _by_id(state, RADAR)["order"] == 7


class TestTheToolPreCheckMatchesTheEndpointRule:
    """The tool's renumber pre-check and this endpoint must agree on ownership.

    ``chat_folder_move`` refuses an app a placement that would renumber a row it
    does not own, and it decides that in the TOOL layer, before its first write —
    because the endpoint judges one row at a time, so a refusal arriving halfway
    leaves the sidebar in an order nobody chose. That means the same rule is
    expressed twice: ``owner_app``-vs-caller in ``mcp_dashboard`` and
    ``_folder_owner_app`` inside this endpoint's ``_apply``.

    These drive the real endpoint rather than a patched ``_patch``, so a change to
    either side's rule — a tightened check, a different absent-key default — turns
    one of them red instead of letting the pre-check quietly permit a write the
    endpoint then refuses (or refuse one it would have allowed).
    """

    @pytest.mark.asyncio
    async def test_the_endpoint_refuses_the_order_write_the_pre_check_refuses(self) -> None:
        """An app writing order on a foreign row: refused, exactly as pre-checked."""
        state = _state(_app_slot("chat-1-200", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"order": 3},
                headers={"X-Session-Key": "dashboard:chat-1-200"},
            )
        assert resp.status == 403
        assert "order" not in (_by_id(state, PERSON) or {})

    @pytest.mark.asyncio
    async def test_the_endpoint_allows_the_order_write_the_pre_check_allows(self) -> None:
        """The same app on its OWN row: allowed, so the pre-check is not over-broad."""
        state = _state(_app_slot("chat-1-200", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"order": 3},
                headers={"X-Session-Key": "dashboard:chat-1-200"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["order"] == 3

    @pytest.mark.asyncio
    async def test_a_legacy_row_reads_as_the_persons_on_both_sides(self) -> None:
        """The absent-key default is the drift the pre-check is most exposed to.

        ``LEGACY`` carries no ``owner_app`` at all. The pre-check reads a missing
        key as the person's via ``.get("owner_app")``; if the endpoint ever read it
        as unowned instead, an app renumber would sail past the pre-check and land.
        """
        state = _state(_app_slot("chat-1-200", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{LEGACY}",
                json={"order": 9},
                headers={"X-Session-Key": "dashboard:chat-1-200"},
            )
        assert resp.status == 403
        assert "order" not in (_by_id(state, LEGACY) or {})


class TestCreateStampsTheOwner:
    @pytest.mark.asyncio
    async def test_an_apps_folder_is_stamped_with_that_app(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 201
        assert body["owner_app"] == "issue-radar"

    @pytest.mark.asyncio
    async def test_the_persons_folder_carries_no_owner_key(self) -> None:
        """Absent, not empty-string: "absent means the person" stays the one
        representation, and the person's rows keep the shape they have on disk."""
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Q3"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 201
        assert "owner_app" not in body

    @pytest.mark.asyncio
    async def test_the_owner_is_never_taken_from_the_body(self) -> None:
        """A caller that could name its own owner could name someone else's."""
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "owner_app": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 201
        assert body["owner_app"] == "issue-radar"

    @pytest.mark.asyncio
    async def test_an_app_may_nest_under_its_own_folder(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "parent_id": RADAR},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 201

    @pytest.mark.asyncio
    async def test_an_app_may_not_nest_under_the_persons_folder(self) -> None:
        """Nesting writes to THAT folder's child list."""
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        before = len(state._folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "parent_id": PERSON},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_not_owned"
        assert len(state._folders) == before

    @pytest.mark.asyncio
    async def test_a_legacy_row_without_the_key_is_the_persons(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "parent_id": LEGACY},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 403


class TestRenameAndReparentAreBounded:
    @pytest.mark.asyncio
    async def test_an_app_can_rename_its_own_folder(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"name": "Renamed"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["name"] == "Renamed"

    @pytest.mark.asyncio
    async def test_an_app_cannot_rename_the_persons_folder(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"name": "Hijacked"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_not_owned"
        assert _by_id(state, PERSON)["name"] == "Work"

    @pytest.mark.asyncio
    async def test_an_app_cannot_rename_another_apps_folder(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{OTHER}",
                json={"name": "Hijacked"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 403
        assert _by_id(state, OTHER)["name"] == "Specs"

    @pytest.mark.asyncio
    async def test_the_person_is_not_confined_by_an_apps_ownership(self) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"name": "Tidied up"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["name"] == "Tidied up"

    @pytest.mark.asyncio
    async def test_an_app_cannot_reparent_its_folder_into_the_persons(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"parent_id": PERSON},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_not_owned"
        assert _by_id(state, RADAR)["parent_id"] == ""

    @pytest.mark.asyncio
    async def test_an_app_can_reparent_to_the_top_level(self) -> None:
        """The top level is not a folder row, so it has no owner to violate —
        that is where an app's own tree starts."""
        folders = _folders()
        nested = {
            "id": "fldr00000005",
            "name": "Runs",
            "parent_id": RADAR,
            "owner_app": "issue-radar",
        }
        folders.append(nested)
        state = _state(_app_slot("chat-1-100", "issue-radar"), folders=folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                "/api/chat/folders/fldr00000005",
                json={"parent_id": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, "fldr00000005")["parent_id"] == ""

    @pytest.mark.asyncio
    async def test_ownership_cannot_be_reassigned_by_a_patch(self) -> None:
        """Stamped once at create; not a field a request can hand over or clear."""
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"owner_app": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["owner_app"] == "issue-radar"

    @pytest.mark.asyncio
    async def test_moving_own_folder_that_holds_a_foreign_one_is_refused(self) -> None:
        """A move takes the subtree with it, so the person's nested folder would
        be relocated by an app's write."""
        folders = _folders()
        folders.append({"id": "fldr00000007", "name": "Theirs", "parent_id": RADAR})
        state = _state(_app_slot("chat-1-100", "issue-radar"), folders=folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"parent_id": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_not_owned"
        assert _by_id(state, RADAR)["parent_id"] == ""

    @pytest.mark.asyncio
    async def test_renaming_a_folder_that_holds_a_foreign_one_is_still_allowed(self) -> None:
        """Only the MOVE is gated on the subtree -- a rename relocates nothing."""
        folders = _folders()
        folders.append({"id": "fldr00000007", "name": "Theirs", "parent_id": RADAR})
        state = _state(_app_slot("chat-1-100", "issue-radar"), folders=folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"name": "Renamed"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["name"] == "Renamed"

    @pytest.mark.asyncio
    async def test_the_person_can_still_move_a_folder_holding_an_apps(self) -> None:
        """Containment cuts both ways, but the person is never confined."""
        folders = _folders()
        folders.append(
            {
                "id": "fldr00000007",
                "name": "Radar sub",
                "parent_id": PERSON,
                "owner_app": "issue-radar",
            }
        )
        state = _state(_ChatSlot("chat-1-100"), folders=folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"parent_id": OTHER},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, PERSON)["parent_id"] == OTHER


class TestAnAppCannotChangeAnExistingFoldersProjectDir:
    """A folder's binding is what every session filed in its subtree picks up
    on its next agent switch, and those sessions live in stores the folder store
    shares no lock with (the slot table, the session archive). "Every session
    under this folder is the caller's own" cannot be established atomically with
    the write, so the PATCH refuses an agent principal's ``project_dir`` change
    outright -- the delete route's rule, on the binding axis. A crew member
    binds a folder at CREATE, when nothing is filed in it yet; an app never
    binds one (``TestAnAppCannotBindAFolderAtAll``); the person keeps the
    update they always had.
    """

    @staticmethod
    def _radar_holding_theirs(bound: str = "") -> list[dict[str, Any]]:
        folders = _folders()
        if bound:
            next(f for f in folders if f["id"] == RADAR)["project_dir"] = bound
        folders.append({"id": "fldr00000007", "name": "Theirs", "parent_id": RADAR})
        return folders

    @pytest.mark.asyncio
    async def test_an_app_cannot_rebind_its_folder_holding_the_persons_chat(self, tmp_path) -> None:
        """The person filed one of their own chats into the app's folder. That
        slot re-resolves the folder's binding on its next agent switch
        (``api_chat_slot_agent``), so an app rebinding the folder would decide
        the person's project, cwd and steering -- cross-ownership through a
        session the folder store cannot see atomically."""
        theirs = _ChatSlot("chat-2-200")
        theirs.folder_id = RADAR
        state = _state(_app_slot("chat-1-100", "issue-radar"), theirs)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert "project_dir" not in _by_id(state, RADAR)

    @pytest.mark.asyncio
    async def test_binding_own_folder_that_holds_a_foreign_one_is_refused(self, tmp_path) -> None:
        """The nested-folder case is one instance of the same rule: a chat the
        person opens in the nested folder would inherit the app's binding."""
        state = _state(_app_slot("chat-1-100", "issue-radar"), folders=self._radar_holding_theirs())
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert "project_dir" not in _by_id(state, RADAR)

    @pytest.mark.asyncio
    async def test_clearing_is_refused_the_same_way(self, tmp_path) -> None:
        """Clearing changes what a filed session picks up just as setting does."""
        state = _state(
            _app_slot("chat-1-100", "issue-radar"),
            folders=self._radar_holding_theirs(bound=str(tmp_path)),
        )
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"project_dir": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 403
        assert _by_id(state, RADAR)["project_dir"] == str(tmp_path)

    @pytest.mark.asyncio
    async def test_even_a_subtree_of_only_its_own_folders_and_sessions_is_refused(
        self, tmp_path
    ) -> None:
        """No narrower rule: an own-folders-only subtree with only the app's own
        live session filed in it is refused too, because the archive (no owner
        in its index, sessions revive with ``folder_id`` intact) and a filing
        that lands mid-request are exactly what the folder store cannot see.
        Nothing is stored and the path is never validated."""
        folders = _folders()
        folders.append(
            {"id": "fldr00000007", "name": "Runs", "parent_id": RADAR, "owner_app": "issue-radar"}
        )
        own = _app_slot("chat-1-100", "issue-radar")
        own.folder_id = RADAR
        state = _state(own, folders=folders)
        with patch("kiro_crew.dashboard.chat_folders._validate_project_dir") as validator:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.patch(
                    f"/api/chat/folders/{RADAR}",
                    json={"project_dir": str(tmp_path)},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
                body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        # An app's refusal names no create-time path: it has none.
        assert body["error"] == (
            "an app cannot set or clear a folder's project directory - ask the person"
        )
        assert "project_dir" not in _by_id(state, RADAR)
        validator.assert_not_called()
        state.mutate_folders.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_refusal_is_audited(self, tmp_path) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        with patch("kiro_crew.dashboard.chat_folders.sel") as sel_fn:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.patch(
                    f"/api/chat/folders/{RADAR}",
                    json={"project_dir": str(tmp_path)},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 403
        kwargs = sel_fn.return_value.log_api_access.call_args.kwargs
        assert kwargs["caller"] == "issue-radar"
        assert kwargs["operation"] == "chat.folder_update"
        assert kwargs["outcome"] == "denied"
        assert kwargs["resources"] == RADAR
        assert "project directory" in kwargs["error"]

    @pytest.mark.asyncio
    async def test_an_apps_other_fields_on_its_own_folder_still_apply(self, tmp_path) -> None:
        """The rule is about the binding only: a rename of the same folder by
        the same app lands as before."""
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"name": "Radar runs"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["name"] == "Radar runs"

    @pytest.mark.asyncio
    async def test_the_person_can_bind_a_folder_holding_an_apps(self, tmp_path) -> None:
        folders = _folders()
        folders.append(
            {
                "id": "fldr00000007",
                "name": "Radar sub",
                "parent_id": PERSON,
                "owner_app": "issue-radar",
            }
        )
        state = _state(_ChatSlot("chat-1-100"), folders=folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, PERSON)["project_dir"] == str(tmp_path.resolve())

    @pytest.mark.asyncio
    async def test_the_path_validator_runs_off_the_event_loop(self, tmp_path) -> None:
        """realpath/isdir on a stalled network path must not hold the gateway
        loop: the PATCH route hands the validator to a worker thread, as the
        create route does."""
        state = _state(_ChatSlot("chat-1-100"))
        seen: list[Any] = []
        real_to_thread = asyncio.to_thread

        async def _spy(fn: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append(fn)
            return await real_to_thread(fn, *args, **kwargs)

        with patch("kiro_crew.dashboard.chat_folders.asyncio.to_thread", side_effect=_spy):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.patch(
                    f"/api/chat/folders/{PERSON}",
                    json={"project_dir": str(tmp_path)},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 200
        # The admission validator (UNC pre-check, then ``_validate_project_dir``)
        # is what the route hands to the worker thread.
        assert _admit_project_dir in seen


class TestAnAppCannotBindAFolderAtAll:
    """An app's ``project_dir`` is agent-chosen host path text that the gateway
    hands to a session as its project, cwd and steering root. With the binding
    accepted at create, an app token could post a private host directory, open
    a slot in the folder and run there. So an app is refused a non-empty
    ``project_dir`` at CREATE as well as on the PATCH (same 403, same code,
    audited under ``app_isolation``, before the path is looked at), and a chat
    filed in an app-owned folder inherits no project directory at all -- the
    read side of the same rule, so a binding that reaches an app's folder by
    any other route (the person's own PATCH, a scaffold) is inert. A crew
    member on a private store keeps the create-time binding, delivered to its
    own chats alone; the person is unconfined.
    """

    @pytest.mark.asyncio
    async def test_an_app_cannot_create_a_bound_folder(self, tmp_path) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        before = len(state._folders)
        with patch("kiro_crew.dashboard.chat_folders._validate_project_dir") as validator:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat/folders",
                    json={"name": "Runs", "project_dir": str(tmp_path)},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
                body = await resp.json()
        assert resp.status == 403, body
        assert body["code"] == "folder_project_dir_forbidden"
        assert "bind it when creating" not in body["error"]
        assert len(state._folders) == before
        validator.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_create_refusal_is_audited_against_the_app(self, tmp_path) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        with patch("kiro_crew.dashboard.chat_folders.sel") as sel_fn:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat/folders",
                    json={"name": "Runs", "project_dir": str(tmp_path)},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 403
        kwargs = sel_fn.return_value.log_api_access.call_args.kwargs
        assert kwargs["caller"] == "issue-radar"
        assert kwargs["operation"] == "chat.folder_create"
        assert kwargs["outcome"] == "denied"
        assert kwargs["source"] == "app_isolation"
        assert "project directory" in kwargs["error"]

    @pytest.mark.asyncio
    async def test_an_app_is_refused_on_the_patch_without_the_create_hint(self, tmp_path) -> None:
        """The PATCH already refused an app; its text pointed the app at
        binding when creating the folder, a path that is now closed to it."""
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        with patch("kiro_crew.dashboard.chat_folders._validate_project_dir") as validator:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.patch(
                    f"/api/chat/folders/{RADAR}",
                    json={"project_dir": str(tmp_path)},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
                body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert "bind it when creating" not in body["error"]
        assert "project_dir" not in _by_id(state, RADAR)
        validator.assert_not_called()
        state.mutate_folders.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_apps_unbound_create_still_lands(self) -> None:
        """Scope pin: the fence is on the BINDING, not on the app's folders."""
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 201, body
        assert body["owner_app"] == "issue-radar"
        assert body["project_dir"] == ""

    def test_a_chat_filed_in_an_app_owned_folder_inherits_no_binding(self, tmp_path) -> None:
        """The read side: whatever put a binding on an app's folder, a chat
        opened there resolves none -- so no write path to an app-owned folder
        can make the gateway launch a session in an app-chosen directory."""
        folders = _folders()
        next(f for f in folders if f["id"] == RADAR)["project_dir"] = str(tmp_path)
        folders.append(
            {"id": "fldr00000007", "name": "Runs", "parent_id": RADAR, "owner_app": "issue-radar"}
        )
        assert _resolve_folder_project_dir(folders, RADAR) == ("", None)
        # An unbound app folder under the bound one: nearest wins would have
        # walked up to the app's binding; the walk starts in an app's folder,
        # so it resolves nothing.
        assert _resolve_folder_project_dir(folders, "fldr00000007") == ("", None)

    def test_the_persons_folder_nested_under_an_apps_bound_folder_inherits_nothing_from_it(
        self, tmp_path
    ) -> None:
        """The binding an app's folder stores is inert for the folders BENEATH it
        too, not only for a chat filed directly in it. The scaffold route writes
        an app-owned folder WITH a binding (its ``project_dir`` is the scanned
        root), so "the ancestor binding can only be the person's" does not hold
        for an app ancestor: a walk that skipped only the start folder handed
        the person's own folder, nested under the app's, the app's directory."""
        folders = _folders()
        next(f for f in folders if f["id"] == RADAR)["project_dir"] = str(tmp_path)
        folders.append({"id": "fldr00000023", "name": "Mine", "parent_id": RADAR, "owner_app": ""})
        assert _resolve_folder_project_dir(folders, "fldr00000023") == ("", None)
        # The move-time walk agrees: a folder placed under the app's inherits
        # no stored binding from it either.
        assert _inherited_project_dir(folders, RADAR) == {}

    def test_an_apps_folder_is_transparent_for_the_persons_own_ancestor_binding(
        self, tmp_path
    ) -> None:
        """Inert, not a wall: the person's folder under an app's folder under
        the person's BOUND folder resolves the person's binding -- the app's
        folder in between is skipped, and the walk goes on to the ancestor the
        person bound. (A chat filed IN the app's folder still resolves nothing:
        the start-folder rule above.)"""
        folders = _folders()
        next(f for f in folders if f["id"] == PERSON)["project_dir"] = str(tmp_path)
        next(f for f in folders if f["id"] == RADAR)["parent_id"] = PERSON
        folders.append({"id": "fldr00000023", "name": "Mine", "parent_id": RADAR, "owner_app": ""})
        assert _resolve_folder_project_dir(folders, "fldr00000023") == (
            str(tmp_path.resolve()),
            None,
        )
        assert _resolve_folder_project_dir(folders, RADAR) == ("", None)
        assert _inherited_project_dir(folders, RADAR) == {"": str(tmp_path)}

    def test_the_persons_binding_reaches_every_chat_and_a_members_only_its_own(
        self, tmp_path
    ) -> None:
        """Owner-scoped delivery, the steering gate's rule: the person's binding
        resolves for every principal's chat; a crew member's resolves for a
        chat running AS that member (``slot_app`` as ``slot_steering_principal``
        spells it) and for nobody else -- not the person's chat filed in the
        member's folder, not another member's, not an app's."""
        folders = _folders()
        next(f for f in folders if f["id"] == PERSON)["project_dir"] = str(tmp_path)
        folders.append(
            {
                "id": "fldr00000008",
                "name": "Reviews",
                "parent_id": "",
                "owner_app": "member:reviewer-store",
                "project_dir": str(tmp_path),
            }
        )
        resolved = str(tmp_path.resolve())
        # The person's chat filed in the member's folder: nothing from it.
        assert _resolve_folder_project_dir(folders, "fldr00000008") == ("", None)
        for principal in ("", "member:reviewer-store", "member:other-store", "issue-radar"):
            assert _resolve_folder_project_dir(folders, PERSON, slot_app=principal) == (
                resolved,
                None,
            )
        assert _resolve_folder_project_dir(
            folders, "fldr00000008", slot_app="member:reviewer-store"
        ) == (resolved, None)
        for principal in ("member:other-store", "issue-radar"):
            assert _resolve_folder_project_dir(folders, "fldr00000008", slot_app=principal) == (
                "",
                None,
            )

    def test_a_members_folder_is_transparent_for_the_persons_own_ancestor_binding(
        self, tmp_path
    ) -> None:
        """Inert, not a wall, for a member's folder as for an app's: the
        person's chat filed in a member-bound folder nested under the person's
        BOUND folder resolves the person's binding -- the member's is skipped
        and the walk goes on. The member's own chat there resolves the member's
        (nearest wins among the bindings that reach it)."""
        folders = _folders()
        next(f for f in folders if f["id"] == PERSON)["project_dir"] = str(tmp_path / "mine")
        (tmp_path / "mine").mkdir()
        (tmp_path / "reviews").mkdir()
        folders.append(
            {
                "id": "fldr00000008",
                "name": "Reviews",
                "parent_id": PERSON,
                "owner_app": "member:reviewer-store",
                "project_dir": str(tmp_path / "reviews"),
            }
        )
        assert _resolve_folder_project_dir(folders, "fldr00000008") == (
            str((tmp_path / "mine").resolve()),
            None,
        )
        assert _resolve_folder_project_dir(
            folders, "fldr00000008", slot_app="member:reviewer-store"
        ) == (str((tmp_path / "reviews").resolve()), None)

    @pytest.mark.asyncio
    async def test_a_crew_member_binds_at_create_for_its_own_chats_only(self, tmp_path) -> None:
        """Pin: the create-time binding is the member's capability -- a
        conductor stands up a project folder for its workers -- and the folder
        it makes is stamped as its own. What the binding confers is
        owner-scoped: the member's own chats filed there start in the project;
        the person's chat filed there from the sidebar inherits nothing, so the
        member's create cannot choose where the PERSON's session is launched."""
        state = _state(_ChatSlot("chat-1-100"))
        member_app = _make_app(state, member_principal="member:reviewer-store")
        async with TestClient(TestServer(member_app)) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Reviews", "project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 201, body
        assert body["owner_app"] == "member:reviewer-store"
        assert body["project_dir"] == str(tmp_path.resolve())
        assert _resolve_folder_project_dir(state._folders, body["id"]) == ("", None)
        assert _resolve_folder_project_dir(
            state._folders, body["id"], slot_app="member:reviewer-store"
        ) == (str(tmp_path.resolve()), None)

    @pytest.mark.asyncio
    async def test_a_crew_member_is_still_refused_on_the_patch_with_the_create_hint(
        self, tmp_path
    ) -> None:
        """Unchanged for a member: an existing folder's binding is the person's,
        and the refusal names the path that IS open to it."""
        own = {
            "id": "fldr00000008",
            "name": "Reviews",
            "parent_id": "",
            "owner_app": "member:reviewer-store",
        }
        state = _state(_ChatSlot("chat-1-100"), folders=[*_folders(), own])
        member_app = _make_app(state, member_principal="member:reviewer-store")
        async with TestClient(TestServer(member_app)) as client:
            resp = await client.patch(
                "/api/chat/folders/fldr00000008",
                json={"project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert "bind it when creating the folder" in body["error"]
        assert "project_dir" not in _by_id(state, "fldr00000008")

    @pytest.mark.asyncio
    async def test_the_person_is_unchanged(self, tmp_path) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            created = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            made = await created.json()
            updated = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert (created.status, updated.status) == (201, 200)
        assert "owner_app" not in made
        assert _by_id(state, PERSON)["project_dir"] == str(tmp_path.resolve())
        assert _resolve_folder_project_dir(state._folders, made["id"]) == (
            str(tmp_path.resolve()),
            None,
        )


class TestAChannelAgentCannotBindAFolder:
    """A Channels agent (session key ``channel:<channel_id>:<agent_id>``) acts on
    words from a thread other people are in. Its key names no dashboard slot and
    no app, so ``folder_principal`` reads it as the PERSON -- and the app/member
    fence on the two binding paths is keyed on that principal. Without a fence
    of its own, a channel agent could bind a new folder or rebind an existing
    one with the person's full authority. Both paths refuse it with the same 403
    the agent-principal fence answers; the person's authority is untouched, and
    a channel agent's other folder writes are not this rule's concern.
    """

    CHANNEL = "channel:chan-000001:helper"

    @pytest.mark.asyncio
    async def test_a_channel_agent_cannot_create_a_bound_folder(self, tmp_path) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        before = len(state._folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "project_dir": str(tmp_path)},
                headers={"X-Session-Key": self.CHANNEL},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert len(state._folders) == before

    @pytest.mark.asyncio
    async def test_a_channel_agent_cannot_change_an_existing_folders_binding(
        self, tmp_path
    ) -> None:
        """Set AND clear: the person's own folder, which a channel key would
        otherwise reach as the person."""
        folders = _folders()
        bound = {"id": "fldr00000008", "name": "Bound", "parent_id": "", "project_dir": "/t"}
        folders.append(bound)
        state = _state(_ChatSlot("chat-1-100"), folders=folders)
        with patch(
            "kiro_crew.dashboard.chat_folders._validate_project_dir",
            return_value=(str(tmp_path), None),
        ) as validator:
            async with TestClient(TestServer(_make_app(state))) as client:
                setting = await client.patch(
                    f"/api/chat/folders/{PERSON}",
                    json={"project_dir": str(tmp_path)},
                    headers={"X-Session-Key": self.CHANNEL},
                )
                set_body = await setting.json()
                clearing = await client.patch(
                    "/api/chat/folders/fldr00000008",
                    json={"project_dir": ""},
                    headers={"X-Session-Key": self.CHANNEL},
                )
        assert (setting.status, clearing.status) == (403, 403)
        assert set_body["code"] == "folder_project_dir_forbidden"
        assert "project_dir" not in _by_id(state, PERSON)
        assert _by_id(state, "fldr00000008")["project_dir"] == "/t"
        # Refused before the path is looked at, like the agent-principal fence.
        validator.assert_not_called()
        state.mutate_folders.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_refusal_is_audited_against_the_channel_key(self, tmp_path) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        with patch("kiro_crew.dashboard.chat_folders.sel") as sel_fn:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat/folders",
                    json={"name": "Runs", "project_dir": str(tmp_path)},
                    headers={"X-Session-Key": self.CHANNEL},
                )
        assert resp.status == 403
        kwargs = sel_fn.return_value.log_api_access.call_args.kwargs
        assert kwargs["caller"] == self.CHANNEL
        assert kwargs["operation"] == "chat.folder_create"
        assert kwargs["outcome"] == "denied"
        assert kwargs["source"] == "channel"
        assert "project directory" in kwargs["error"]

    @pytest.mark.asyncio
    async def test_a_channel_agents_unbound_folder_writes_are_not_this_rule(self, tmp_path) -> None:
        """Scope pin: the fence is on the BINDING. An unbound create and a
        rename by the same channel key land exactly as they did."""
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            created = await client.post(
                "/api/chat/folders",
                json={"name": "Runs"},
                headers={"X-Session-Key": self.CHANNEL},
            )
            renamed = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"name": "Work items"},
                headers={"X-Session-Key": self.CHANNEL},
            )
        assert (created.status, renamed.status) == (201, 200)
        assert _by_id(state, PERSON)["name"] == "Work items"

    @pytest.mark.asyncio
    async def test_the_person_still_binds_at_create_and_update(self, tmp_path) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            created = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            updated = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert (created.status, updated.status) == (201, 200)
        assert _by_id(state, PERSON)["project_dir"] == str(tmp_path.resolve())


class TestAMoveCannotChangeWhatASubtreeInherits:
    """The composition the create-only rule leaves open: a crew member binds a
    NEW folder at create (allowed -- nothing is filed in it yet), then
    reparents an EXISTING folder it owns under it. Every session filed in the
    moved subtree -- including one of the person's the folder store cannot see
    -- resolves the destination's binding on its next agent switch, so the move
    rebinds them exactly as the refused PATCH would have. The same holds in the
    other direction (moving out from under a binding clears it). The fence is
    keyed on the binding principal and compares what the two places resolve
    to, so an APP's move is exempt: an app's subtree holds only its own folders
    (``foreign_descendant``) and a chat filed in an app's folder resolves no
    binding wherever the folder sits, so no move of the app's can change what
    any chat resolves -- refusing it would refuse a harmless reorganisation.

    Rule: an agent principal's reparent may not change what ANY chat in the
    moved subtree resolves, compared per principal -- delivery is owner-scoped,
    so a place confers one binding on the person's chats and possibly another
    on a member's own. A folder the PERSON bound moves freely (its binding
    reaches every chat, so its subtree resolves it first wherever it sits); a
    folder a crew member bound at create does not (its binding stops only the
    member's own chats -- the person's chat filed in it resolves the ancestors
    -- so the move is compared for everyone else); an unbound one may move only
    between places that confer the same bindings. The person is never confined.
    """

    MEMBER = "member:reviewer-store"
    REVIEWS = "fldr00000011"
    BOUND = "fldr00000009"
    BOUND_CHILD = "fldr00000010"

    @staticmethod
    def _tree(bound_dir: str, reviews_parent: str = "") -> list[dict[str, Any]]:
        cls = TestAMoveCannotChangeWhatASubtreeInherits
        folders = _folders()
        folders.append(
            {
                "id": cls.REVIEWS,
                "name": "Reviews",
                "parent_id": reviews_parent,
                "owner_app": cls.MEMBER,
            }
        )
        folders.append(
            {
                "id": cls.BOUND,
                "name": "Bound at create",
                "parent_id": "",
                "owner_app": cls.MEMBER,
                "project_dir": bound_dir,
            }
        )
        folders.append(
            {
                "id": cls.BOUND_CHILD,
                "name": "Under bound",
                "parent_id": cls.BOUND,
                "owner_app": cls.MEMBER,
            }
        )
        return folders

    def _member_app(self, state: DashboardState) -> web.Application:
        return _make_app(state, member_principal=self.MEMBER)

    @pytest.mark.asyncio
    async def test_a_member_cannot_move_its_folder_under_one_it_bound_at_create(
        self, tmp_path
    ) -> None:
        """The exact composition: create C with project_dir, then reparent the
        member's existing folder -- holding one of the person's chats and the
        member's own -- under C. Before: the move lands and the member's own
        chats filed there resolve C's directory on their next agent switch (the
        person's never resolve a member's binding). After: refused with the
        binding-fence code, nothing inherited by anyone."""
        theirs = _ChatSlot("chat-2-200")
        theirs.folder_id = self.REVIEWS
        state = _state(_ChatSlot("chat-1-100"), theirs, folders=self._tree(str(tmp_path)))
        async with TestClient(TestServer(self._member_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{self.REVIEWS}",
                json={"parent_id": self.BOUND},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert _by_id(state, self.REVIEWS)["parent_id"] == ""
        # What the person's chat resolves on its next agent switch: still nothing.
        assert _resolve_folder_project_dir(state._folders, self.REVIEWS) == ("", None)

    @pytest.mark.asyncio
    async def test_moving_out_from_under_a_binding_is_refused_the_same_way(self, tmp_path) -> None:
        """The clear direction: the subtree would stop inheriting."""
        state = _state(
            _ChatSlot("chat-1-100"),
            folders=self._tree(str(tmp_path), reviews_parent=self.BOUND),
        )
        async with TestClient(TestServer(self._member_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{self.REVIEWS}",
                json={"parent_id": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert _by_id(state, self.REVIEWS)["parent_id"] == self.BOUND

    @pytest.mark.asyncio
    async def test_a_move_that_keeps_the_inherited_binding_still_lands(self, tmp_path) -> None:
        """Between two places under the same bound ancestor nothing changes for
        the subtree, so the member's own tree stays organisable."""
        state = _state(
            _ChatSlot("chat-1-100"),
            folders=self._tree(str(tmp_path), reviews_parent=self.BOUND),
        )
        async with TestClient(TestServer(self._member_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{self.REVIEWS}",
                json={"parent_id": self.BOUND_CHILD},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, self.REVIEWS)["parent_id"] == self.BOUND_CHILD

    @pytest.mark.asyncio
    async def test_a_member_bound_folder_moves_where_nobody_elses_binding_changes(
        self, tmp_path
    ) -> None:
        """The member's own binding stops its own chats wherever the folder
        sits, and neither place confers anything on anyone else (both are
        member-bound or unbound), so the move changes nothing for any chat and
        lands."""
        folders = self._tree(str(tmp_path))
        next(f for f in folders if f["id"] == self.REVIEWS)["project_dir"] = str(tmp_path / "own")
        state = _state(_ChatSlot("chat-1-100"), folders=folders)
        async with TestClient(TestServer(self._member_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{self.REVIEWS}",
                json={"parent_id": self.BOUND},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, self.REVIEWS)["parent_id"] == self.BOUND

    @pytest.mark.asyncio
    async def test_a_member_bound_folder_cannot_leave_the_persons_binding(self, tmp_path) -> None:
        """Owner-scoped delivery makes a member-bound folder NOT exempt: its
        binding reaches only the member's own chats, so the person's chat filed
        in it resolves the ancestors -- here the person's bound folder -- and
        moving the folder to the top level would clear that chat's project.
        Refused like any other crossing; the member's own chats are unaffected
        either way (they resolve the folder's own binding)."""
        folders = self._tree(str(tmp_path), reviews_parent=PERSON)
        next(f for f in folders if f["id"] == PERSON)["project_dir"] = str(tmp_path / "mine")
        (tmp_path / "mine").mkdir()
        (tmp_path / "own").mkdir()
        next(f for f in folders if f["id"] == self.REVIEWS)["project_dir"] = str(tmp_path / "own")
        theirs = _ChatSlot("chat-2-200")
        theirs.folder_id = self.REVIEWS
        state = _state(_ChatSlot("chat-1-100"), theirs, folders=folders)
        assert _resolve_folder_project_dir(state._folders, self.REVIEWS) == (
            str((tmp_path / "mine").resolve()),
            None,
        )
        async with TestClient(TestServer(self._member_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{self.REVIEWS}",
                json={"parent_id": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403, body
        assert body["code"] == "folder_project_dir_forbidden"
        assert _by_id(state, self.REVIEWS)["parent_id"] == PERSON
        assert _resolve_folder_project_dir(state._folders, self.REVIEWS) == (
            str((tmp_path / "mine").resolve()),
            None,
        )
        assert _resolve_folder_project_dir(state._folders, self.REVIEWS, slot_app=self.MEMBER) == (
            str((tmp_path / "own").resolve()),
            None,
        )

    @pytest.mark.asyncio
    async def test_the_refusal_is_audited(self, tmp_path) -> None:
        state = _state(_ChatSlot("chat-1-100"), folders=self._tree(str(tmp_path)))
        with patch("kiro_crew.dashboard.chat_folders.sel") as sel_fn:
            async with TestClient(TestServer(self._member_app(state))) as client:
                resp = await client.patch(
                    f"/api/chat/folders/{self.REVIEWS}",
                    json={"parent_id": self.BOUND},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 403
        kwargs = sel_fn.return_value.log_api_access.call_args.kwargs
        assert kwargs["caller"] == self.MEMBER
        assert kwargs["operation"] == "chat.folder_update"
        assert kwargs["outcome"] == "denied"
        assert kwargs["resources"] == self.REVIEWS
        assert "inherit" in kwargs["error"]

    @pytest.mark.asyncio
    async def test_an_apps_move_across_a_binding_lands_because_nothing_resolved_changes(
        self, tmp_path
    ) -> None:
        """An app's folder, holding one of the person's chats, sits under the
        person's BOUND folder (the person nested it there). The chat resolves no
        binding -- it is filed in an app's folder -- and would resolve none at
        the top level either, so the app's move out from under the binding is
        not a rebind and lands. Compared as stored strings alone it would have
        read as a crossing (the person's directory, then nothing); the fence
        compares what the places RESOLVE to for the moved subtree."""
        folders = _folders()
        next(f for f in folders if f["id"] == PERSON)["project_dir"] = str(tmp_path)
        next(f for f in folders if f["id"] == RADAR)["parent_id"] = PERSON
        theirs = _ChatSlot("chat-2-200")
        theirs.folder_id = RADAR
        state = _state(_app_slot("chat-1-100", "issue-radar"), theirs, folders=folders)
        assert _resolve_folder_project_dir(state._folders, RADAR) == ("", None)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"parent_id": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 200, body
        assert _by_id(state, RADAR)["parent_id"] == ""
        assert _resolve_folder_project_dir(state._folders, RADAR) == ("", None)

    @pytest.mark.asyncio
    async def test_the_person_moves_across_bindings_freely(self, tmp_path) -> None:
        state = _state(_ChatSlot("chat-1-100"), folders=self._tree(str(tmp_path)))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{self.REVIEWS}",
                json={"parent_id": self.BOUND},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, self.REVIEWS)["parent_id"] == self.BOUND


class TestAChannelAgentsMoveIsFencedOnTheBindingToo:
    """The third path a channel agent reaches as the person. Its key names no
    slot and no app, so ``folder_principal`` is ``""`` for it and the ownership
    branches of the PATCH do not confine it -- a channel agent may reparent the
    PERSON's folders, as it always could. The cross-binding move rule must not
    key on that same principal, or the one caller the create and set-or-clear
    fences refuse walks through the reparent: moving the person's unbound folder
    under a bound one rebinds every chat filed in it on its next agent switch,
    exactly as the refused PATCH would. One binding principal, computed once per
    request, keys all three fences; the ownership branches stay on the folder
    principal, so a channel agent's same-binding moves land as before.
    """

    CHANNEL = "channel:chan-000001:helper"
    BOUND = "fldr00000011"
    BOUND_CHILD = "fldr00000012"

    @staticmethod
    def _tree(bound_dir: str, person_parent: str = "") -> list[dict[str, Any]]:
        folders = _folders()
        next(f for f in folders if f["id"] == PERSON)["parent_id"] = person_parent
        # The PERSON's bound folder: no owner key, so a channel agent reaches it
        # exactly as it reaches the rest of the person's tree.
        folders.append(
            {
                "id": TestAChannelAgentsMoveIsFencedOnTheBindingToo.BOUND,
                "name": "Bound",
                "parent_id": "",
                "project_dir": bound_dir,
            }
        )
        folders.append(
            {
                "id": TestAChannelAgentsMoveIsFencedOnTheBindingToo.BOUND_CHILD,
                "name": "Under bound",
                "parent_id": TestAChannelAgentsMoveIsFencedOnTheBindingToo.BOUND,
            }
        )
        return folders

    @pytest.mark.asyncio
    async def test_a_channel_agent_cannot_move_the_persons_folder_under_a_binding(
        self, tmp_path
    ) -> None:
        """Before: the move lands and the person's chat filed in the folder
        resolves the bound folder's directory on its next agent switch. After:
        refused with the binding-fence code, nothing inherited."""
        theirs = _ChatSlot("chat-2-200")
        theirs.folder_id = PERSON
        state = _state(_ChatSlot("chat-1-100"), theirs, folders=self._tree(str(tmp_path)))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"parent_id": self.BOUND},
                headers={"X-Session-Key": self.CHANNEL},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert _by_id(state, PERSON)["parent_id"] == ""
        assert _resolve_folder_project_dir(state._folders, PERSON) == ("", None)

    @pytest.mark.asyncio
    async def test_moving_out_from_under_a_binding_is_refused_the_same_way(self, tmp_path) -> None:
        """The clear direction: the person's chats inside would stop inheriting."""
        state = _state(
            _ChatSlot("chat-1-100"), folders=self._tree(str(tmp_path), person_parent=self.BOUND)
        )
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"parent_id": ""},
                headers={"X-Session-Key": self.CHANNEL},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert _by_id(state, PERSON)["parent_id"] == self.BOUND

    @pytest.mark.asyncio
    async def test_a_move_that_keeps_the_inherited_binding_still_lands(self, tmp_path) -> None:
        """Scope pin: the fence is on the BINDING, not on the channel agent's
        other folder writes -- a move between two places under the same bound
        ancestor changes nothing for the subtree and lands as it always did."""
        state = _state(
            _ChatSlot("chat-1-100"), folders=self._tree(str(tmp_path), person_parent=self.BOUND)
        )
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"parent_id": self.BOUND_CHILD},
                headers={"X-Session-Key": self.CHANNEL},
            )
        assert resp.status == 200
        assert _by_id(state, PERSON)["parent_id"] == self.BOUND_CHILD

    @pytest.mark.asyncio
    async def test_the_refusal_is_audited_against_the_channel_key(self, tmp_path) -> None:
        state = _state(_ChatSlot("chat-1-100"), folders=self._tree(str(tmp_path)))
        with patch("kiro_crew.dashboard.chat_folders.sel") as sel_fn:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.patch(
                    f"/api/chat/folders/{PERSON}",
                    json={"parent_id": self.BOUND},
                    headers={"X-Session-Key": self.CHANNEL},
                )
        assert resp.status == 403
        kwargs = sel_fn.return_value.log_api_access.call_args.kwargs
        assert kwargs["caller"] == self.CHANNEL
        assert kwargs["operation"] == "chat.folder_update"
        assert kwargs["outcome"] == "denied"
        assert kwargs["source"] == "channel"
        assert kwargs["resources"] == PERSON
        assert "inherit" in kwargs["error"]


class TestASessionDrivenFromAMessagingChannelIsFencedTheSameWay:
    """A turn driven from a messaging-transport thread -- a ``slack:`` thread, a
    ``discord:`` DM, any namespace ``messaging.link.is_channel_session_key``
    classifies -- carries that key as its verified ``X-Session-Key``. Like a
    Channels agent's key it names no slot and no app, so ``folder_principal``
    reads it as the PERSON; unlike it, it does not start with ``channel:``, so a
    binding principal that recognised only that prefix let the same words from
    a thread other people are in bind a folder, rebind one, move one across a
    binding and (in the steering module) declare host-file reads with the
    person's authority. The principal now recognises the transport namespaces
    through the repo's one predicate -- the roster is imported here, not
    re-listed -- so every fence keyed on it confines these keys too. The same
    key's writes that are not a binding (an unbound create, a rename, a move
    between places with the same binding) land as before.
    """

    SLACK = "slack:1785370133.085469"
    BOUND = "fldr00000021"
    BOUND_CHILD = "fldr00000022"

    @staticmethod
    def _tree(bound_dir: str, person_parent: str = "") -> list[dict[str, Any]]:
        folders = _folders()
        next(f for f in folders if f["id"] == PERSON)["parent_id"] = person_parent
        folders.append(
            {
                "id": TestASessionDrivenFromAMessagingChannelIsFencedTheSameWay.BOUND,
                "name": "Bound",
                "parent_id": "",
                "project_dir": bound_dir,
            }
        )
        folders.append(
            {
                "id": TestASessionDrivenFromAMessagingChannelIsFencedTheSameWay.BOUND_CHILD,
                "name": "Under bound",
                "parent_id": TestASessionDrivenFromAMessagingChannelIsFencedTheSameWay.BOUND,
            }
        )
        return folders

    @pytest.mark.asyncio
    @pytest.mark.parametrize("namespace", CHANNEL_SESSION_NAMESPACES)
    async def test_no_transport_namespace_can_create_a_bound_folder(
        self, tmp_path, namespace: str
    ) -> None:
        """Every namespace the roster names, not one spelling: the fence reuses the
        predicate, so a transport added to the roster is fenced without a change here."""
        state = _state(_ChatSlot("chat-1-100"))
        before = len(state._folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "project_dir": str(tmp_path)},
                headers={"X-Session-Key": f"{namespace}:conversation-1"},
            )
            body = await resp.json()
        assert resp.status == 403, body
        assert body["code"] == "folder_project_dir_forbidden"
        assert len(state._folders) == before

    @pytest.mark.asyncio
    async def test_it_cannot_set_or_clear_an_existing_folders_binding(self, tmp_path) -> None:
        state = _state(_ChatSlot("chat-1-100"), folders=self._tree("/t"))
        with patch(
            "kiro_crew.dashboard.chat_folders._validate_project_dir",
            return_value=(str(tmp_path), None),
        ) as validator:
            async with TestClient(TestServer(_make_app(state))) as client:
                setting = await client.patch(
                    f"/api/chat/folders/{PERSON}",
                    json={"project_dir": str(tmp_path)},
                    headers={"X-Session-Key": self.SLACK},
                )
                set_body = await setting.json()
                clearing = await client.patch(
                    f"/api/chat/folders/{self.BOUND}",
                    json={"project_dir": ""},
                    headers={"X-Session-Key": self.SLACK},
                )
        assert (setting.status, clearing.status) == (403, 403), set_body
        assert set_body["code"] == "folder_project_dir_forbidden"
        assert "project_dir" not in _by_id(state, PERSON)
        assert _by_id(state, self.BOUND)["project_dir"] == "/t"
        validator.assert_not_called()
        state.mutate_folders.assert_not_called()

    @pytest.mark.asyncio
    async def test_it_cannot_move_the_persons_folder_across_a_binding(self, tmp_path) -> None:
        """Both directions: under a binding (the person's chat filed inside would
        inherit it) and out from under one (it would stop inheriting)."""
        theirs = _ChatSlot("chat-2-200")
        theirs.folder_id = PERSON
        state = _state(_ChatSlot("chat-1-100"), theirs, folders=self._tree(str(tmp_path)))
        async with TestClient(TestServer(_make_app(state))) as client:
            under = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"parent_id": self.BOUND},
                headers={"X-Session-Key": self.SLACK},
            )
            under_body = await under.json()
        assert under.status == 403, under_body
        assert under_body["code"] == "folder_project_dir_forbidden"
        assert _by_id(state, PERSON)["parent_id"] == ""
        assert _resolve_folder_project_dir(state._folders, PERSON) == ("", None)

        state = _state(
            _ChatSlot("chat-1-100"), folders=self._tree(str(tmp_path), person_parent=self.BOUND)
        )
        async with TestClient(TestServer(_make_app(state))) as client:
            out = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"parent_id": ""},
                headers={"X-Session-Key": self.SLACK},
            )
        assert out.status == 403
        assert _by_id(state, PERSON)["parent_id"] == self.BOUND

    @pytest.mark.asyncio
    async def test_the_refusal_is_audited_against_the_transport_key(self, tmp_path) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        with patch("kiro_crew.dashboard.chat_folders.sel") as sel_fn:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat/folders",
                    json={"name": "Runs", "project_dir": str(tmp_path)},
                    headers={"X-Session-Key": self.SLACK},
                )
        assert resp.status == 403
        kwargs = sel_fn.return_value.log_api_access.call_args.kwargs
        assert kwargs["caller"] == self.SLACK
        assert kwargs["operation"] == "chat.folder_create"
        assert kwargs["outcome"] == "denied"
        assert kwargs["source"] == "channel"

    @pytest.mark.asyncio
    async def test_its_unbound_writes_and_same_binding_moves_still_land(self, tmp_path) -> None:
        """Scope pin: the fence is on the BINDING. An unbound create, a rename and
        a move between two places under the same bound ancestor land exactly as
        they did for a Slack-driven turn."""
        state = _state(
            _ChatSlot("chat-1-100"), folders=self._tree(str(tmp_path), person_parent=self.BOUND)
        )
        async with TestClient(TestServer(_make_app(state))) as client:
            created = await client.post(
                "/api/chat/folders", json={"name": "Runs"}, headers={"X-Session-Key": self.SLACK}
            )
            renamed = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"name": "Tidied"},
                headers={"X-Session-Key": self.SLACK},
            )
            moved = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"parent_id": self.BOUND_CHILD},
                headers={"X-Session-Key": self.SLACK},
            )
        assert (created.status, renamed.status, moved.status) == (201, 200, 200)
        assert _by_id(state, PERSON)["parent_id"] == self.BOUND_CHILD


class TestADashboardSlotLinkedToAThreadIsAChannelCallerToo:
    """The third shape of the same caller. ``/kirocrew link-to-dashboard`` links a
    Slack thread to a DASHBOARD-BORN slot: the allowed users in that thread then
    drive its turns (``slack.handler.maybe_route_linked_thread`` runs them with
    channel provenance), but the slot's session key stays ``dashboard:<slot>``,
    so a principal derived from the key alone read those turns as the PERSON --
    words from a thread other people are in bound folders, rebound them and
    declared host-file reads. The link is gateway-owned slot state
    (``state.link_slack`` is its only writer, persisted through the session
    map), so the binding principal reads it: a linked slot's caller key resolves
    to the thread's own ``slack:<ts>`` key, and every fence keyed on the
    principal confines it exactly as it confines a ``slack:`` session. The same
    slot unlinked is the person again; writes that are not a binding land.
    """

    THREAD = "1785370133.085469"

    @staticmethod
    def _linked_slot() -> _ChatSlot:
        slot = _ChatSlot("chat-1-100")
        slot._slack_linked = True
        slot._slack_thread_ts = TestADashboardSlotLinkedToAThreadIsAChannelCallerToo.THREAD
        slot._slack_channel = "C0000000001"
        return slot

    @pytest.mark.asyncio
    async def test_a_linked_slot_cannot_create_a_bound_folder(self, tmp_path) -> None:
        state = _state(self._linked_slot())
        before = len(state._folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403, body
        assert body["code"] == "folder_project_dir_forbidden"
        assert "channel agent" in body["error"]
        assert len(state._folders) == before

    @pytest.mark.asyncio
    async def test_a_linked_slot_cannot_set_an_existing_folders_binding(self, tmp_path) -> None:
        state = _state(self._linked_slot())
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403, body
        assert body["code"] == "folder_project_dir_forbidden"
        assert "project_dir" not in _by_id(state, PERSON)

    @pytest.mark.asyncio
    async def test_a_linked_slot_cannot_declare_steering(self, tmp_path) -> None:
        """The steering fence keys on the same principal, so it confines the
        linked slot without a rule of its own."""
        state = _state(self._linked_slot())
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "steering_dirs": [str(tmp_path)]},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403, body
        assert body["code"] == "steering_dirs_forbidden"

    @pytest.mark.asyncio
    async def test_the_refusal_is_audited_against_the_threads_channel_key(self, tmp_path) -> None:
        state = _state(self._linked_slot())
        with patch("kiro_crew.dashboard.chat_folders.sel") as sel_fn:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat/folders",
                    json={"name": "Runs", "project_dir": str(tmp_path)},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 403
        kwargs = sel_fn.return_value.log_api_access.call_args.kwargs
        assert kwargs["caller"] == f"slack:{self.THREAD}"
        assert kwargs["source"] == "channel"
        assert kwargs["outcome"] == "denied"

    @pytest.mark.asyncio
    async def test_the_same_slot_unlinked_is_the_person(self, tmp_path) -> None:
        """The contrast that pins WHAT is read: the link, not the slot's name or
        its transcript. Unlinked, the same key binds at create as the person."""
        slot = self._linked_slot()
        slot._slack_linked = False
        state = _state(slot)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 201, body
        assert body["project_dir"] == str(tmp_path.resolve())

    @pytest.mark.asyncio
    async def test_a_linked_slots_unbound_writes_are_not_this_rule(self) -> None:
        """The link confines the binding and the steering declaration, nothing
        else: an unbound create and a rename land as the person's."""
        state = _state(self._linked_slot())
        async with TestClient(TestServer(_make_app(state))) as client:
            created = await client.post(
                "/api/chat/folders",
                json={"name": "Runs"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            renamed = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"name": "Tidied"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert (created.status, renamed.status) == (201, 200)
        assert _by_id(state, PERSON)["name"] == "Tidied"

    @staticmethod
    def _unlinked_mid_turn() -> _ChatSlot:
        """A slot whose thread-produced turn is still RUNNING after the link was
        cleared (the unlink route, or a thread handoff to another slot): the
        turn's own origin snapshot, published by ``_run_chat`` at its start,
        survives; the live link does not."""
        slot = _ChatSlot("chat-1-100")
        slot._active_turn_channel_origin = (
            f"slack:{TestADashboardSlotLinkedToAThreadIsAChannelCallerToo.THREAD}"
        )
        slot._slack_linked = False
        slot._slack_thread_ts = ""
        slot._slack_channel = ""
        return slot

    @pytest.mark.asyncio
    async def test_a_turn_the_thread_produced_stays_a_channel_caller_after_an_unlink(
        self, tmp_path
    ) -> None:
        """The principal is read off the RUNNING TURN's origin before the link.
        The link is mutable underneath a turn -- the unlink route and a thread
        handoff both clear it while the thread's turn keeps running -- and a
        principal read off the live link alone relabelled that turn's later
        calls as the person's, so words from the thread bound a folder with the
        person's authority the moment the person (or a handoff) unlinked the
        slot mid-turn. The snapshot ``_run_chat`` publishes at turn start
        answers instead, and the refusal is audited against the thread."""
        state = _state(self._unlinked_mid_turn())
        before = len(state._folders)
        with patch("kiro_crew.dashboard.chat_folders.sel") as sel_fn:
            async with TestClient(TestServer(_make_app(state))) as client:
                created = await client.post(
                    "/api/chat/folders",
                    json={"name": "Runs", "project_dir": str(tmp_path)},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
                created_body = await created.json()
                patched = await client.patch(
                    f"/api/chat/folders/{PERSON}",
                    json={"project_dir": str(tmp_path)},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
                steering = await client.post(
                    "/api/chat/folders",
                    json={"name": "Runs", "steering_dirs": [str(tmp_path)]},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert created.status == 403, created_body
        assert created_body["code"] == "folder_project_dir_forbidden"
        assert patched.status == 403
        assert steering.status == 403
        assert len(state._folders) == before
        assert "project_dir" not in _by_id(state, PERSON)
        kwargs = sel_fn.return_value.log_api_access.call_args_list[0].kwargs
        assert kwargs["caller"] == f"slack:{self.THREAD}"
        assert kwargs["source"] == "channel"

    @pytest.mark.asyncio
    async def test_the_turns_origin_outranks_the_live_link(self, tmp_path) -> None:
        """Order pin: a running thread-produced turn is answered from its own
        snapshot even when the slot has since been linked to a DIFFERENT thread
        (a handoff the other way), so the audit names the thread whose words
        are running, not the one that now owns the slot."""
        slot = self._linked_slot()
        slot._active_turn_channel_origin = "slack:1785370133.000001"
        state = _state(slot)
        with patch("kiro_crew.dashboard.chat_folders.sel") as sel_fn:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat/folders",
                    json={"name": "Runs", "project_dir": str(tmp_path)},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 403
        assert sel_fn.return_value.log_api_access.call_args.kwargs["caller"] == (
            "slack:1785370133.000001"
        )

    @pytest.mark.asyncio
    async def test_a_dashboard_produced_turn_on_an_unlinked_slot_is_the_person(
        self, tmp_path
    ) -> None:
        """The contrast that pins the snapshot's SCOPE: a running turn the
        dashboard produced publishes no channel origin (``""``), so with no link
        either the same key binds as the person -- the snapshot confines a
        channel's turn, never the person's own."""
        slot = self._linked_slot()
        slot._slack_linked = False
        slot._active_turn_channel_origin = ""
        state = _state(slot)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 201, body


class TestAMoveCannotChangeWhatASubtreeInheritsForSteeringEither:
    """The second thing a folder's ancestry decides for every chat filed beneath
    it: the steering directories it inherits, ACCUMULATIVELY up ``parent_id``
    (``_resolve_folder_steering_dirs``), read into each chat's model context at
    session start. The binding branch of the move rule compared only the
    inherited ``project_dir``, so a channel agent -- refused a ``steering_dirs``
    declaration of its own -- could reparent the person's folder under one that
    declares steering whenever both places inherit the same binding, and the
    person's chats filed inside picked those documents up at their next start:
    the declaration fence, reached through the tree. So the same pre-commit
    branch compares what the moved subtree would inherit for steering too --
    the stored declarations of every ancestor, root-first, with each declaring
    folder's owner, exactly the data the resolver consumes -- and refuses a
    move that changes it, in either direction, with the steering fence's own
    code. A move between two places that inherit the same steering lands; the
    person is not confined.
    """

    CHANNEL = "channel:chan-000001:helper"
    STEERED = "fldr00000031"
    STEERED_CHILD = "fldr00000032"
    RADAR_STEERED = "fldr00000033"
    RADAR_LEAF = "fldr00000034"

    @staticmethod
    def _tree(person_parent: str = "") -> list[dict[str, Any]]:
        folders = _folders()
        next(f for f in folders if f["id"] == PERSON)["parent_id"] = person_parent
        cls = TestAMoveCannotChangeWhatASubtreeInheritsForSteeringEither
        folders.extend(
            [
                # The PERSON's folder declaring steering: stored strings, never
                # validated here -- the branch runs under the store lock.
                {
                    "id": cls.STEERED,
                    "name": "Standards",
                    "parent_id": "",
                    "steering_dirs": ["/srv/standards"],
                },
                {"id": cls.STEERED_CHILD, "name": "Under standards", "parent_id": cls.STEERED},
                # An app's folder on which the PERSON declared steering (allowed:
                # the delivery gate routes it to the app's own chats), holding an
                # unbound folder of the app's.
                {
                    "id": cls.RADAR_STEERED,
                    "name": "Radar standards",
                    "parent_id": "",
                    "owner_app": "issue-radar",
                    "steering_dirs": ["/srv/radar-standards"],
                },
                {
                    "id": cls.RADAR_LEAF,
                    "name": "Radar runs",
                    "parent_id": cls.RADAR_STEERED,
                    "owner_app": "issue-radar",
                },
            ]
        )
        return folders

    @pytest.mark.asyncio
    async def test_a_channel_agent_cannot_move_the_persons_folder_under_declared_steering(
        self,
    ) -> None:
        """Both places inherit the same (empty) binding, so the binding branch is
        silent. Before: the move lands and the person's chat filed inside
        inherits ``/srv/standards`` at its next start. After: refused with the
        steering fence's code, nothing inherited."""
        theirs = _ChatSlot("chat-2-200")
        theirs.folder_id = PERSON
        state = _state(_ChatSlot("chat-1-100"), theirs, folders=self._tree())
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"parent_id": self.STEERED},
                headers={"X-Session-Key": self.CHANNEL},
            )
            body = await resp.json()
        assert resp.status == 403, body
        assert body["code"] == "steering_dirs_forbidden"
        assert _by_id(state, PERSON)["parent_id"] == ""
        assert _inherited_steering_dirs(state._folders, PERSON) == ()

    @pytest.mark.asyncio
    async def test_moving_out_from_under_declared_steering_is_refused_too(self) -> None:
        """The clear direction: the person's chats inside would stop receiving
        the standards the person put above them."""
        state = _state(_ChatSlot("chat-1-100"), folders=self._tree(person_parent=self.STEERED))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"parent_id": ""},
                headers={"X-Session-Key": self.CHANNEL},
            )
            body = await resp.json()
        assert resp.status == 403, body
        assert body["code"] == "steering_dirs_forbidden"
        assert _by_id(state, PERSON)["parent_id"] == self.STEERED

    @pytest.mark.asyncio
    async def test_an_app_is_held_to_the_same_rule_on_its_own_folders(self) -> None:
        """Its own unbound folder, out from under its own folder on which the
        person declared steering: the app's chats filed in it would stop
        receiving what the person declared for them. Refused, as every agent
        principal is."""
        state = _state(_app_slot("chat-1-100", "issue-radar"), folders=self._tree())
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{self.RADAR_LEAF}",
                json={"parent_id": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403, body
        assert body["code"] == "steering_dirs_forbidden"
        assert _by_id(state, self.RADAR_LEAF)["parent_id"] == self.RADAR_STEERED

    @pytest.mark.asyncio
    async def test_a_move_that_keeps_the_inherited_steering_still_lands(self) -> None:
        """Scope pin: between two places under the same declaring ancestor the
        accumulated set is identical, so the move lands as it always did."""
        state = _state(_ChatSlot("chat-1-100"), folders=self._tree(person_parent=self.STEERED))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"parent_id": self.STEERED_CHILD},
                headers={"X-Session-Key": self.CHANNEL},
            )
        assert resp.status == 200, await resp.text()
        assert _by_id(state, PERSON)["parent_id"] == self.STEERED_CHILD

    @pytest.mark.asyncio
    async def test_the_person_is_not_confined(self) -> None:
        state = _state(_ChatSlot("chat-1-100"), folders=self._tree())
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"parent_id": self.STEERED},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200, await resp.text()
        assert _by_id(state, PERSON)["parent_id"] == self.STEERED

    @pytest.mark.asyncio
    async def test_the_refusal_is_audited_against_the_channel_key(self) -> None:
        state = _state(_ChatSlot("chat-1-100"), folders=self._tree())
        with patch("kiro_crew.dashboard.chat_folders.sel") as sel_fn:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.patch(
                    f"/api/chat/folders/{PERSON}",
                    json={"parent_id": self.STEERED},
                    headers={"X-Session-Key": self.CHANNEL},
                )
        assert resp.status == 403
        kwargs = sel_fn.return_value.log_api_access.call_args.kwargs
        assert kwargs["caller"] == self.CHANNEL
        assert kwargs["operation"] == "chat.folder_update"
        assert kwargs["outcome"] == "denied"
        assert kwargs["source"] == "channel"
        assert kwargs["resources"] == PERSON
        assert "steering" in kwargs["error"]


class TestAnAppCannotDeleteFolders:
    """A delete relocates everything the folder contains, and those contents live
    in a DIFFERENT store from the folder -- the slot table and the session
    archive, neither sharing a lock with it. So emptiness cannot be established
    atomically with the removal, and every narrower rule leaked through another
    seam. The person keeps the delete they always had.

    Nothing shipped loses a capability: no MCP tool exposes folder deletion, and
    the only client of the route is the dashboard UI.
    """

    @pytest.mark.asyncio
    async def test_an_app_cannot_delete_even_an_empty_folder_it_owns(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        with patch("kiro_crew.dashboard.chat_folders.save_slot_off_loop", AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.delete(
                    f"/api/chat/folders/{RADAR}",
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
                body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_delete_forbidden"
        assert _by_id(state, RADAR) is not None

    @pytest.mark.asyncio
    async def test_no_session_is_touched_by_the_refusal(self) -> None:
        """Refused before the unfile loop, so nothing is written and there is
        nothing to roll back."""
        mine = _app_slot("chat-9-900", "issue-radar")
        mine.folder_id = RADAR
        state = _state(_app_slot("chat-1-100", "issue-radar"), mine)
        with patch("kiro_crew.dashboard.chat_folders.save_slot_off_loop", AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.delete(
                    f"/api/chat/folders/{RADAR}",
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 403
        assert mine.folder_id == RADAR

    @pytest.mark.asyncio
    async def test_the_person_can_still_delete_a_full_folder(self) -> None:
        """The person is not confined: clearing out a folder full of
        conversations and subfolders is the delete they already had."""
        theirs = _app_slot("chat-9-900", "issue-radar")
        theirs.folder_id = RADAR
        folders = _folders()
        folders.append({"id": "fldr00000006", "name": "Sub", "parent_id": RADAR})
        state = _state(_ChatSlot("chat-1-100"), theirs, folders=folders)
        with patch("kiro_crew.dashboard.chat_folders.save_slot_off_loop", AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.delete(
                    f"/api/chat/folders/{RADAR}",
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 200
        assert _by_id(state, RADAR) is None
        assert theirs.folder_id == ""
        assert _by_id(state, "fldr00000006")["parent_id"] == ""


class TestACallerWhoseSlotIsGoneIsRefused:
    """An empty scope reads as the person, which is right for a caller that never
    had a slot (Slack, a channel session, the person's cron) and wrong for a
    `dashboard:` key, which NAMES one. A tab closing while its tool call is in
    flight pops the slot without draining, so an app-owned session would arrive
    unattributable and be handed the person's authority over the person's folders.
    """

    @pytest.mark.asyncio
    async def test_create_is_refused(self) -> None:
        state = _state()  # the named slot is absent from the registry
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Sneaky"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "caller_unattributable"

    @pytest.mark.asyncio
    async def test_rename_of_the_persons_folder_is_refused(self) -> None:
        state = _state()
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"name": "Hijacked"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 403
        assert _by_id(state, PERSON)["name"] == "Work"

    @pytest.mark.asyncio
    async def test_delete_is_refused_before_any_slot_is_unfiled(self) -> None:
        filed = _ChatSlot("chat-9-900")
        filed.folder_id = PERSON
        state = _state(filed)
        with patch("kiro_crew.dashboard.chat_folders.save_slot_off_loop", AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.delete(
                    f"/api/chat/folders/{PERSON}",
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 403
        assert _by_id(state, PERSON) is not None
        assert filed.folder_id == PERSON

    @pytest.mark.asyncio
    async def test_a_caller_that_never_had_a_slot_is_still_the_person(self) -> None:
        """The refusal must not swallow Slack, channel or cron callers -- they
        never had a slot to be confined to, which is a different fact."""
        state = _state()
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"name": "Tidied"},
                headers={"X-Session-Key": "slack:T1/C1"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["name"] == "Tidied"
