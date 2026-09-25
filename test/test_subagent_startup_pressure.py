"""Tests for the two startup-pressure guards (subagent.py, admission/pump.py,
monitoring.py).

A fan-out is bounded by three different quantities. ``_max_concurrent`` bounds
how many agents RUN; ``_spawn_stagger_secs`` bounds the RATE of starts; and --
new here -- ``_startup_cap`` bounds how many admitted agents are IN STARTUP at
once (executing or parked at the spawn-approval prompt; no runtime PID, no
first provider stream, no turn -- plus a ``ClaimPoint`` reservation not yet
registered). Without the third bound one start was admitted per interval however long each
took, and under slow starts a wide fan-out piled dozens of agents into startup
together; the fixed 120s startup watchdog then reaped healthy starts as
``Failed to start within 120s`` (measured ~50% loss on a 120-item wave against
~2% at 24-45 items).

Guard A: ``_should_stagger_queue`` and the drain pump hold further spawns in the
EXISTING queue while the in-startup population is at ``_startup_cap``; the
queue wakes on a PID / first stream (``_note_startup_progress``) and on the
slot-release drain of a terminal, including the watchdog's reap of a wedged
start, so a wedged population cannot hold the queue past its reap.

The watchdog's deadline itself stays the fixed ``_startup_deadline`` whatever the
crowd; what moves is the clock, reset at ``SessionStartGate`` exit on both start
paths (``_gate_exit_reset``). A pressure-aware deadline was tried here and
withdrawn -- see ``TestWatchdogIgnoresTheCrowd`` for the invariant that replaced
it -- which is also what keeps the single-agent contract of
``test_subagent_startup_watchdog.py`` unchanged.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from overload_fakes import mock_ctx, mock_sessions

# One spelling across the suite for a pid no platform can allocate: a fake pid a
# live process could own reaches whatever holds it on the runner when a cleanup
# path signals it (under pytest-xdist, a sibling worker).
from test_update_provider import _UNALLOCATABLE_PID

import kiro_crew.subagent as subagent_mod
from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.subagent import _STARTUP_CAP_GATE_ROUNDS, SubagentInfo, SubagentManager

# The end-to-end tests drive ``SubagentManager.spawn``, which refuses on a
# memory-pressured host; pin the host reading so the verdict is the test's.
pytestmark = pytest.mark.usefixtures("healthy_host_memory")

# ── helpers ───────────────────────────────────────────────────────────────


def _manager(
    *, max_concurrent: int = 8, startup_timeout: int = 120, gate_width: int = 2
) -> SubagentManager:
    """A manager whose in-startup bound is ``2 x gate_width`` (clamped to the cap).

    The bound has no config key -- it is derived from the session-start gate's
    width alone -- so a test that wants a bound of 2 asks for a gate of 1.
    """
    mgr = SubagentManager(
        sessions=mock_sessions(),
        ctx_builder=mock_ctx(),
        max_concurrent=max_concurrent,
        startup_timeout=startup_timeout,
    )
    mgr._session_start_concurrency = gate_width
    return mgr


def _info(agent_id: str = "a1b2c3d4", **overrides) -> SubagentInfo:
    info = SubagentInfo(id=agent_id, task="t", agent="")
    for k, v in overrides.items():
        setattr(info, k, v)
    return info


def _starting(agent_id: str, exec_started: float | None = 100.0, **overrides) -> SubagentInfo:
    """An agent in startup: executing, but nothing to show for it yet."""
    return _info(agent_id, **{"_exec_started": exec_started, "_pid": None, "turns": 0, **overrides})


def _register(mgr: SubagentManager, *infos: SubagentInfo) -> None:
    for info in infos:
        mgr._agents[info.id] = info


# ── _in_startup: admitted with nothing to show yet ────────────────────────


class TestInStartupPredicate:
    def test_executing_with_nothing_to_show_is_in_startup(self) -> None:
        assert SubagentManager._in_startup(_starting("s1")) is True

    def test_approval_parked_is_in_startup_because_it_is_admitted(self) -> None:
        """An agent parked at the spawn-approval prompt holds an admission and
        goes straight to ``_run`` on approval with no further gate, so the
        bound must count it while it is parked -- that is the ONLY moment the
        bound can act on it. The watchdog stays blind to it (no clock yet)."""
        parked = _info(_exec_started=None, turns=0, _awaiting_approval=True)
        assert SubagentManager._in_startup(parked) is True
        mgr = _manager()
        assert mgr._is_startup_stalled(parked, now=1e9) is False

    def test_a_reservation_counts_until_its_info_is_registered(self) -> None:
        """A durable-store spawn reserves its slot before its claim is awaited
        and registers only on re-entry, which skips the gate; the reservation
        stands in for the missing info so admissions decided in between see it."""
        mgr = _manager(max_concurrent=8, gate_width=1)  # bound 2
        mgr._spawn_stagger_secs = 0.0
        _register(mgr, _starting("a"))
        assert mgr._should_stagger_queue(time.monotonic())[0] is False
        mgr._startup_reservations = 1
        assert mgr._startup_population() == 2
        assert mgr._should_stagger_queue(time.monotonic())[0] is True
        mgr._admission.release_reservation("r")
        assert mgr._startup_reservations == 0
        assert mgr._should_stagger_queue(time.monotonic())[0] is False

    @pytest.mark.parametrize(
        "leaving",
        [
            {"_pid": _UNALLOCATABLE_PID},
            {"_first_stream_started": 101.0},
            # A mid-run TOOL prompt sets the same flag; the stream/turn it
            # arrived on keeps it out of the population.
            {"_first_stream_started": 101.0, "turns": 1, "_awaiting_approval": True},
            {"turns": 1},
            {"done": True},
            {"_reap_started": True},
        ],
    )
    def test_every_way_out_of_startup_leaves_the_population(self, leaving: dict) -> None:
        assert SubagentManager._in_startup(_starting("s1", **leaving)) is False

    def test_population_counts_only_registered_agents_in_startup(self) -> None:
        mgr = _manager()
        a, b, c = _starting("a"), _starting("b"), _starting("c", 50.0)
        c._pid = _UNALLOCATABLE_PID
        _register(mgr, a, b, c)
        assert mgr._startup_population() == 2
        assert mgr._startup_population(exclude=a) == 1
        # An unregistered info excludes nothing.
        assert mgr._startup_population(exclude=_starting("x")) == 2


# ── _startup_cap: configured, or derived from the running cap ─────────────


class TestStartupCap:
    @pytest.mark.parametrize(
        ("cap", "ssc", "expected"),
        [
            (8, 2, 4),  # 2 x gate width; the cap does not enter
            (40, 2, 4),
            (64, 2, 4),  # was 16 under the withdrawn ceil(cap / 4) term: 7 gate rounds queued
            (64, 4, 8),  # a wider gate admits a wider round
            (8, 4, 8),  # 2 x 4 = 8, clamped to the cap
            (3, 2, 3),  # 2 x 2 = 4 clamped to the cap of 3
            (1, 2, 1),
            (0, 2, 1),  # an adaptive squeeze to 0: the running cap pauses admission, not this
        ],
    )
    def test_derives_two_gate_rounds_clamped_to_the_cap(
        self, cap: int, ssc: int, expected: int
    ) -> None:
        mgr = _manager(max_concurrent=max(cap, 1), gate_width=ssc)
        mgr._max_concurrent = cap
        assert _STARTUP_CAP_GATE_ROUNDS == 2
        assert mgr._startup_cap() == expected

    def test_derived_bound_does_not_grow_with_the_cap(self) -> None:
        """The reap the bound prevents is produced at the gate, so the bound is
        a function of the gate's width alone: raising the cap must not queue
        more rounds of pre-permit wait behind the same permits."""
        mgr = _manager(max_concurrent=8, gate_width=2)
        bounds = []
        for cap in (8, 16, 40, 64, 128):
            mgr._max_concurrent = cap
            bounds.append(mgr._startup_cap())
        assert bounds == [4, 4, 4, 4, 4]

    def test_there_is_no_config_key_for_the_bound(self) -> None:
        """``2 x gate width`` is both floor and ceiling of the useful range, so
        an override could only make the bound worse; the operator's lever is
        ``agent.session_start_concurrency``, which the bound tracks. The
        earlier ``agent.subagent_max_concurrent_startups`` key is gone from the
        schema, the live-config list and the manager."""
        assert not hasattr(KiroCrewConfig().agent, "subagent_max_concurrent_startups")
        assert "agent.subagent_max_concurrent_startups" not in SubagentManager.LIVE_CONFIG_PATHS
        mgr = _manager(max_concurrent=40, gate_width=2)
        assert not hasattr(mgr, "_max_concurrent_startups_setting")
        # apply_limits leaves the derivation alone: a cap change re-clamps, nothing else.
        cfg = KiroCrewConfig()
        cfg.agent.max_subagents = 40
        mgr.apply_limits(cfg, max_concurrent=40)
        assert mgr._startup_cap() == 4
        mgr.apply_limits(cfg, max_concurrent=3)
        assert mgr._startup_cap() == 3


# ── Guard A at the gate: the third clause of _should_stagger_queue ────────


class TestGateHoldsAtTheStartupCap:
    def test_a_free_slot_is_still_held_while_startup_is_full(self) -> None:
        mgr = _manager(max_concurrent=8, gate_width=1)  # bound 2
        mgr._spawn_stagger_secs = 0.0
        _register(mgr, _starting("a"), _starting("b"))
        should_queue, slot_free = mgr._should_stagger_queue(time.monotonic())
        # Held -- and ``slot_free`` still tells the truth about the cap, so
        # the caller knows no running agent's exit is what will release this.
        assert (should_queue, slot_free) == (True, True)

    def test_one_agent_leaving_startup_opens_the_gate(self) -> None:
        mgr = _manager(max_concurrent=8, gate_width=1)  # bound 2
        mgr._spawn_stagger_secs = 0.0
        a, b = _starting("a"), _starting("b")
        _register(mgr, a, b)
        a._pid = _UNALLOCATABLE_PID
        should_queue, slot_free = mgr._should_stagger_queue(time.monotonic())
        assert (should_queue, slot_free) == (False, True)

    def test_a_wedged_agent_that_was_reaped_no_longer_counts(self) -> None:
        mgr = _manager(max_concurrent=8, gate_width=1)  # bound 2
        mgr._spawn_stagger_secs = 0.0
        wedged = _starting("w")
        _register(mgr, wedged, _starting("peer"))
        assert mgr._should_stagger_queue(time.monotonic())[0] is True
        wedged._reap_started = True  # the reaper's first write, before any await
        assert mgr._should_stagger_queue(time.monotonic())[0] is False


# ── Guard A end to end: the queue holds, wakes on progress, drains on a reap ─
#
# Every path from admission to ``_run`` is exercised here: the auto-approved
# ones (yolo / approval_mode="auto" / parent trust / hooks) start inside
# ``spawn()`` right after the gate, and the interactive one parks in
# ``_spawn_with_approval`` and enters ``_run`` when its prompt resolves -- the
# path a bulk trust/yolo grant releases all at once.


class _StartupRuns:
    """Patch ``_run`` with a run that ENTERS startup and then waits for the test.

    Each run marks ``_exec_started`` -- the real ``_run_inner``'s first
    statement -- and parks on a future; the test moves it out of startup
    (``progress``: a PID, the way a real start does) or ends it (``finish``).
    """

    def __init__(self) -> None:
        self.parked: dict[str, asyncio.Future] = {}
        self.started: list[str] = []
        runs = self

        async def _run(mgr: SubagentManager, info: SubagentInfo) -> None:
            info._exec_started = time.time()
            runs.started.append(info.id)
            fut = runs.parked.setdefault(info.id, asyncio.get_event_loop().create_future())
            await fut
            info.done = True
            info.result = "ok"
            mgr._claim_finalize(info)
            if mgr._release_slot(info):
                mgr._running_count -= 1
                mgr._drain_queue()

        self._patches = [patch.object(SubagentManager, "_run", new=_run)]

    def __enter__(self) -> "_StartupRuns":
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc: object) -> None:
        for p in self._patches:
            p.stop()

    def progress(
        self, mgr: SubagentManager, info: SubagentInfo, pid: int = _UNALLOCATABLE_PID
    ) -> None:
        live = mgr._agents[info.id]
        live._pid = pid
        mgr._note_startup_progress(live)

    async def finish(self, mgr: SubagentManager, info: SubagentInfo) -> None:
        fut = self.parked.setdefault(info.id, asyncio.get_event_loop().create_future())
        if not fut.done():
            fut.set_result("ok")
        await _settle()


async def _settle(rounds: int = 25) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


async def _spawn(mgr: SubagentManager, task: str) -> SubagentInfo:
    # One tick between admissions, as the stagger timer guarantees in
    # production: the admitted run's first step (``_exec_started``) lands
    # before the next spawn reads the population.
    info = mgr.spawn(task, parent_session_key="dash:pressure")
    assert info is not None
    await _settle()
    return info


async def _close(mgr: SubagentManager, runs: _StartupRuns) -> None:
    mgr._shutting_down = True
    for fut in runs.parked.values():
        if not fut.done():
            fut.set_result("ok")
    tasks = [t for t in mgr._tasks.values() if not t.done()]
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    mgr._taskq.close()


@pytest.mark.asyncio
async def test_queue_holds_at_the_startup_cap_and_resumes_on_progress() -> None:
    """Cap 8, gate width 1 (bound 2), four spawns: two start, two wait in the
    EXISTING queue -- no second queue -- and each leaves as one agent in
    startup records a PID."""
    with _StartupRuns() as runs:
        mgr = _manager(max_concurrent=8, gate_width=1)
        mgr._spawn_stagger_secs = 0.0
        await mgr.wait_taskq_ready()
        try:
            first = await _spawn(mgr, "one")
            second = await _spawn(mgr, "two")
            third = await _spawn(mgr, "three")
            fourth = await _spawn(mgr, "four")

            assert runs.started == [first.id, second.id]
            assert third.queued and not third.done
            assert fourth.queued and not fourth.done
            assert mgr._startup_population() == 2
            assert mgr._running_count == 2  # the cap had six free slots; startup held them
            assert len(mgr._queue) == 2

            runs.progress(mgr, first)  # a runtime PID: out of startup
            await _settle()
            assert runs.started == [first.id, second.id, third.id]
            assert mgr._startup_population() == 2  # second + third
            assert len(mgr._queue) == 1

            runs.progress(mgr, second)
            await _settle()
            assert runs.started == [first.id, second.id, third.id, fourth.id]
            assert not mgr._queue
        finally:
            await _close(mgr, runs)


@pytest.mark.asyncio
async def test_bulk_approval_cannot_release_more_than_the_startup_cap() -> None:
    """The bulk-approval path, driven end to end through ``spawn()`` and the
    REAL ``_spawn_with_approval``: cap 8, gate width 1 (bound 2), four spawns
    that each need a human prompt. Two are admitted and park at the prompt;
    the other two must wait in the queue -- a parked agent is admitted and will
    enter startup with no further gate, so it counts against the bound while
    parked. Then every pending prompt is resolved in one pass, exactly as
    ``tool_approval:bulk_trust`` / ``bulk_yolo`` does, and the grant turns
    trust on: only the two parked agents enter startup, and the queue releases
    the rest one at a time as startup progresses. Before this bound counted
    parked agents, all four parked (none "in startup") and all four entered
    ``_run`` together on the grant -- the wide-fan-out pile-up under bulk
    approval that the measurement came from."""
    pending: dict[str, asyncio.Future] = {}
    trusted = False

    async def _prompt(request_id: str, description: str, parent_session_key: str = "") -> bool:
        if trusted:
            return True
        fut = asyncio.get_event_loop().create_future()
        pending[request_id] = fut
        return await fut

    with _StartupRuns() as runs:
        mgr = _manager(max_concurrent=8, gate_width=1)
        mgr._spawn_stagger_secs = 0.0
        # Interactive approval only: no yolo, no parent trust, no hook auto-approve.
        mgr._sessions.get_approval_policy = MagicMock(return_value="")
        mgr._ctx_builder.hooks.auto_approve_subagent_spawn = False
        mgr._on_spawn_approval = _prompt
        await mgr.wait_taskq_ready()
        try:
            infos = [await _spawn(mgr, f"prompted-{i}") for i in range(4)]
            first, second, third, fourth = infos
            assert runs.started == []  # nobody past the prompt yet
            assert sorted(pending) == sorted(f"spawn:{i.id}" for i in (first, second))
            assert mgr._agents[first.id]._awaiting_approval is True
            assert third.queued and not third.done
            assert fourth.queued and not fourth.done
            assert mgr._startup_population() == 2  # the two parked at the prompt
            assert len(mgr._queue) == 2

            # The bulk grant: trust flips on and every pending prompt resolves at once.
            trusted = True
            for fut in list(pending.values()):
                fut.set_result(True)
            await _settle()
            assert runs.started == [first.id, second.id]  # not four
            assert mgr._startup_population() == 2
            assert third.queued and fourth.queued
            assert len(mgr._queue) == 2

            runs.progress(mgr, first)  # a runtime PID: out of startup, one slot opens
            await _settle()
            assert runs.started == [first.id, second.id, third.id]
            assert mgr._startup_population() == 2  # second + third
            runs.progress(mgr, second)
            await _settle()
            assert runs.started == [first.id, second.id, third.id, fourth.id]
            assert not mgr._queue
        finally:
            await _close(mgr, runs)


@pytest.mark.asyncio
async def test_queue_drains_after_the_watchdog_reaps_a_wedged_start() -> None:
    """Gate width 1 (bound 2), two wedged starts and a waiter: the wedged
    pair holds the queue only until the watchdog reaps one of them -- the
    reap's slot-release drain admits the waiter, so the in-startup count
    cannot stick and deadlock the queue."""
    with _StartupRuns() as runs:
        mgr = _manager(max_concurrent=8, gate_width=1, startup_timeout=120)
        mgr._spawn_stagger_secs = 0.0
        await mgr.wait_taskq_ready()
        # Neuter the reap's process/tombstone collaborators, keep its queue drain.
        mgr._sessions.reset = AsyncMock()
        mgr._sigkill_session = AsyncMock()  # type: ignore[method-assign]
        mgr._write_tombstone = MagicMock()  # type: ignore[method-assign]
        mgr._record_cost = MagicMock()  # type: ignore[method-assign]
        mgr._fire_event = AsyncMock()  # type: ignore[method-assign]
        try:
            wedged = await _spawn(mgr, "wedged")
            other = await _spawn(mgr, "other")
            waiter = await _spawn(mgr, "waiter")
            assert runs.started == [wedged.id, other.id]
            assert waiter.queued and not waiter.done

            live = mgr._agents[wedged.id]
            now = live._exec_started + 121.0
            assert mgr._is_startup_stalled(live, now) is True
            await mgr._force_reap(wedged.id, live, 121.0, reason="startup_timeout")
            await _settle()

            assert live.done is True
            assert "Failed to start within 120s" in live.error
            assert mgr._startup_population() == 2  # other + the waiter, now starting
            assert runs.started == [wedged.id, other.id, waiter.id]
            assert not mgr._queue
        finally:
            await _close(mgr, runs)


# ── The watchdog deadline is fixed; the crowd is diagnostics only ─────────
#
# A pressure-aware deadline (base plus a term per OTHER agent in startup) was
# tried on this branch and withdrawn. The measurement that motivated the work
# predates the gate-exit reset and the in-startup bound, so nothing showed a
# healthy start still missing the BASE deadline once queue time stopped being
# charged; and the term as written was not monotonic -- sampled at sweep time
# against ``now - _exec_started``, which spans the whole crowded period, it
# shrank as the crowd drained, so an agent inside its window at one sweep
# could be reaped at the next. These tests pin the invariant that replaced it.


class TestWatchdogIgnoresTheCrowd:
    def test_lone_wedged_agent_is_reaped_at_the_base_deadline(self) -> None:
        mgr = _manager(startup_timeout=120)
        lone = _starting("lone", exec_started=1_000.0)
        _register(mgr, lone)
        assert mgr._is_startup_stalled(lone, now=1_000.0 + 120.0) is False
        assert mgr._is_startup_stalled(lone, now=1_000.0 + 120.5) is True

    def test_a_crowd_buys_a_wedged_agent_no_extra_time(self) -> None:
        """Four peers in startup: the deadline is still the base, not more."""
        mgr = _manager(startup_timeout=120)
        wedged = _starting("wedged", exec_started=1_000.0)
        _register(mgr, wedged, *[_starting(f"p{i}", 1_000.0) for i in range(4)])
        assert mgr._is_startup_stalled(wedged, now=1_000.0 + 120.0) is False
        assert mgr._is_startup_stalled(wedged, now=1_000.0 + 120.5) is True

    def test_a_draining_crowd_cannot_shorten_a_window_already_granted(self) -> None:
        """The non-monotonicity the withdrawn term had: inside the window at one
        sweep with peers present, the same agent must still be inside it at the
        next sweep after every peer has left."""
        mgr = _manager(startup_timeout=120)
        me = _starting("me", exec_started=1_000.0)
        peers = [_starting(f"p{i}", 1_000.0) for i in range(4)]
        _register(mgr, me, *peers)
        assert mgr._is_startup_stalled(me, now=1_000.0 + 119.0) is False
        for peer in peers:
            peer._pid = _UNALLOCATABLE_PID  # every peer leaves startup
        assert mgr._startup_population(exclude=me) == 0
        assert mgr._is_startup_stalled(me, now=1_000.0 + 119.5) is False
        assert mgr._is_startup_stalled(me, now=1_000.0 + 120.5) is True


@pytest.mark.asyncio
async def test_reaper_sweep_reaps_at_the_base_deadline_under_a_crowd(monkeypatch) -> None:
    """One real sweep of ``_reaper_loop`` with two agents in startup: the one
    past the base deadline is reaped under ``startup_timeout``; the one inside
    it -- under the same crowd -- is left alone. The reaper's warning names the
    fixed deadline and the in-startup population, and the error names the fixed
    deadline."""
    mgr = _manager(startup_timeout=120)
    wedged = _starting("wedged", exec_started=1_000.0)
    fresh = _starting("fresh", exec_started=1_000.0 + 100.0)
    _register(mgr, wedged, fresh)
    reaped: list[tuple[str, str]] = []
    swept = asyncio.Event()

    async def _force_reap(agent_id, info, elapsed, *, reason=""):
        reaped.append((agent_id, reason))
        info.done = True
        swept.set()

    mgr._force_reap = _force_reap  # type: ignore[method-assign]
    mgr._refresh_learned_cost = MagicMock()  # type: ignore[method-assign]
    mgr._rebuild_conversation_registry = AsyncMock()  # type: ignore[method-assign]
    mgr._sample_live_costs = MagicMock()  # type: ignore[method-assign]
    mgr._sweep_stuck_waves_async = AsyncMock()  # type: ignore[method-assign]
    mgr._sweep_digest_holds_async = AsyncMock()  # type: ignore[method-assign]
    mgr._sweep_conversations = MagicMock()  # type: ignore[method-assign]
    mgr._taskq_pump = MagicMock()  # type: ignore[method-assign]
    mgr._maybe_flag_stall = AsyncMock()  # type: ignore[method-assign]
    monkeypatch.setattr(subagent_mod, "_REAPER_INTERVAL", 0)
    monkeypatch.setattr(subagent_mod, "compact_cost_log", lambda: None)
    monkeypatch.setattr(subagent_mod, "prune_stale_tombstones", lambda *a, **k: 0)
    # +125s: past wedged's base deadline (one peer in startup would have bought
    # it 135s under the withdrawn term), inside fresh's own window.
    monkeypatch.setattr(
        subagent_mod,
        "time",
        SimpleNamespace(time=lambda: 1_000.0 + 125.0, monotonic=time.monotonic),
    )

    loop_task = asyncio.ensure_future(mgr._reaper_loop())
    try:
        await asyncio.wait_for(swept.wait(), 5.0)
    finally:
        loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)
        mgr._taskq.close()

    assert reaped == [("wedged", "startup_timeout")]
    assert fresh.done is False


# ── Gate-exit start-clock reset on the DEDICATED-process path ─────────────
#
# ``_create_shared_session`` already hands ``runtime.create_session`` an
# ``on_gate_acquired`` that resets ``_exec_started`` at ``SessionStartGate``
# exit, so time parked at the gate is not charged to the startup deadline. The
# dedicated path (``get_or_create`` -- every ``model`` / ``reasoning_effort``
# spawn) had no equivalent: the same reset now rides ``get_or_create`` ->
# provider factory -> ``AcpProvider`` -> ``create_session``. ONE definition
# (``_gate_exit_reset``) serves both paths.


class TestGateExitResetIsOneDefinition:
    def test_reset_moves_the_start_clock_and_records_the_wait(self, monkeypatch) -> None:
        mgr = _manager()
        info = _starting("a1", exec_started=100.0)
        info.last_activity = 100.0
        monkeypatch.setattr(subagent_mod, "time", SimpleNamespace(time=lambda: 250.0))
        reset = mgr._gate_exit_reset(info)
        reset(150_000.0)
        assert info._exec_started == 250.0
        assert info.last_activity == 250.0
        assert info._start_queue_wait_ms == 150_000.0

    def test_the_watchdog_measures_from_gate_exit_not_admission(self) -> None:
        """Gate wait is admission's cost: a start that queued 200s and then
        began is judged from the moment it began."""
        mgr = _manager(startup_timeout=120)
        info = _starting("a1", exec_started=1_000.0)
        _register(mgr, info)
        # Without the reset, 200s past _exec_started reaps it.
        assert mgr._is_startup_stalled(info, now=1_000.0 + 200.0) is True
        with patch.object(subagent_mod, "time", SimpleNamespace(time=lambda: 1_200.0)):
            mgr._gate_exit_reset(info)(200_000.0)
        assert mgr._is_startup_stalled(info, now=1_200.0 + 100.0) is False
        # A dedicated start wedged AFTER it holds its permit is still reaped at
        # the base deadline, counted from gate exit.
        assert mgr._is_startup_stalled(info, now=1_200.0 + 120.5) is True

    @pytest.mark.asyncio
    async def test_shared_path_uses_the_same_reset(self, tmp_path) -> None:
        """The shared path's callback IS ``_gate_exit_reset``'s: one convention."""
        mgr = _manager()
        runtime = MagicMock()
        runtime.create_session = AsyncMock()
        mgr._sessions.get_subagent_runtime = AsyncMock(return_value=runtime)
        mgr._get_parent_runtime = MagicMock(return_value=None)  # type: ignore[method-assign]
        mgr._bind_shared_handle = AsyncMock()  # type: ignore[method-assign]
        info = _starting("shared", exec_started=100.0)
        info.parent_session_key = "dash:p"
        _register(mgr, info)
        await mgr._create_shared_session(info, "subagent:shared", "")
        cb = runtime.create_session.await_args.kwargs["on_gate_acquired"]
        with patch.object(subagent_mod, "time", SimpleNamespace(time=lambda: 777.0)):
            cb(5.0)
        assert info._exec_started == 777.0 and info._start_queue_wait_ms == 5.0


class TestDedicatedPathGateExitReset:
    """The dedicated path threads the reset through ``get_or_create``."""

    @staticmethod
    def _dedicated_sessions(captured: dict) -> MagicMock:
        provider = AsyncMock()
        provider.start = AsyncMock()
        provider.shutdown = AsyncMock()
        provider.context_usage_pct = lambda: 0.0
        provider.context_used_tokens = MagicMock(return_value=0)
        provider.context_window_tokens = MagicMock(return_value=0)

        async def stream(*_a, **_k):
            from kiro_crew.providers.base import EVENT_COMPLETE

            yield SimpleNamespace(kind=EVENT_COMPLETE, stop_reason="end_turn", runtime_global=False)

        provider.stream = MagicMock(side_effect=stream)

        async def get_or_create(*_args, **kwargs):
            captured.update(kwargs)
            return provider, True, False

        sessions = mock_sessions()
        sessions.get_or_create = AsyncMock(side_effect=get_or_create)
        sessions.record_success = MagicMock()
        return sessions

    @pytest.mark.asyncio
    async def test_run_inner_hands_get_or_create_the_gate_exit_reset(self, monkeypatch) -> None:
        from kiro_crew.execution_context import execution_for_store
        from kiro_crew.subagent_persistence import create_agent_folder

        captured: dict = {}
        sessions = self._dedicated_sessions(captured)
        ctx = MagicMock()
        ctx.build_message = MagicMock(return_value=("message", None))
        ctx.hooks.auto_approve_subagent_tools = False
        mgr = SubagentManager(sessions=sessions, ctx_builder=ctx)
        mgr._should_use_session_sharing = MagicMock(return_value=False)  # type: ignore[method-assign]
        info = _info("d1d2d3d4", execution_context=execution_for_store(""))
        info.model = "gpt-5.6-sol"  # a model pin: the dedicated path by decision
        await asyncio.to_thread(
            create_agent_folder, info.id, execution_context=info.execution_context
        )

        await mgr._run_inner(info, "subagent:d1d2d3d4")

        reset = captured.get("on_gate_acquired")
        assert callable(reset), captured.keys()
        with patch.object(subagent_mod, "time", SimpleNamespace(time=lambda: 4_321.0)):
            reset(90_000.0)
        assert info._exec_started == 4_321.0
        assert info._start_queue_wait_ms == 90_000.0

    def test_provider_factory_names_the_kwarg_and_forwards_it(self, monkeypatch) -> None:
        """Named in ``_acp``, never swallowed by its ``**_kwargs`` catch-all."""
        import kiro_crew.providers.acp as acp_mod

        captured: list[dict] = []

        class _FakeProvider:
            def __init__(self, **kwargs: object) -> None:
                captured.append(kwargs)

        monkeypatch.setattr(acp_mod, "AcpProvider", _FakeProvider)
        factory = KiroCrewConfig().create_provider_factory()
        marker = lambda _ms: None  # noqa: E731
        factory("subagent:x", agent=None, on_gate_acquired=marker)
        factory("dash:y", agent=None)
        assert captured[0]["on_gate_acquired"] is marker
        assert captured[1]["on_gate_acquired"] is None

    @pytest.mark.asyncio
    async def test_acp_provider_forwards_the_reset_to_its_own_create_session(self) -> None:
        """The dedicated process's ``session/new`` gets the callback; a
        ``session/load`` resume takes no gate permit and gets none."""
        from kiro_crew.providers.acp import AcpProvider

        marker = lambda _ms: None  # noqa: E731
        with patch("kiro_crew.providers.acp.AcpClient"):
            provider = AcpProvider(acp_backend="", on_gate_acquired=marker)
        provider._client = MagicMock()
        provider._client.backend = ""
        provider._client._work_dir = "/tmp/ws"
        provider._client._agent = "kirocrew"
        provider._client._sandbox_mode = "auto"
        provider._client._extra_env = {}
        provider._client._mcp_gateway_overlay = None
        provider._client._mcp_gateway_socket = None
        provider._client._model = "auto"
        provider._client._resume_session_id = ""
        handle = MagicMock()
        handle.session_id = "kiro-sess-1"
        handle.store_session_config = MagicMock()
        handle.set_model = AsyncMock()
        runtime = MagicMock()
        runtime.pid = _UNALLOCATABLE_PID
        runtime.spawn = AsyncMock()
        runtime.create_session = AsyncMock(return_value=handle)
        runtime.load_session = AsyncMock(return_value=handle)
        with (
            patch("kiro_crew.providers.acp.AcpRuntime", return_value=runtime),
            patch(
                "kiro_crew.providers.acp.AcpSessionProvider",
                side_effect=lambda h, r, **kw: MagicMock(_handle=h, _runtime=r, resumed=False),
            ),
        ):
            await provider._start_kiro_runtime()
        runtime.create_session.assert_awaited_once()
        assert runtime.create_session.await_args.kwargs["on_gate_acquired"] is marker
        runtime.load_session.assert_not_awaited()

    def test_a_provider_built_without_the_callback_passes_none(self) -> None:
        from kiro_crew.providers.acp import AcpProvider

        with patch("kiro_crew.providers.acp.AcpClient"):
            provider = AcpProvider(acp_backend="")
        assert provider._on_gate_acquired is None
