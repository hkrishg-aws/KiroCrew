"""A mirrored name's owner has ONE storage location, in every module that mirrors one.

``sys.modules`` is where a module is stored. A module that mirrors a surface and
also holds a mapping to the resolved owner MODULE has a second storage location
for it, and the two can disagree: purging an owner and importing it again -- an
idiom this suite uses in twenty files -- leaves such a mapping reading and
forwarding writes to the discarded module while a direct importer holds the fresh
one. A test patching a control through the mirror then passes while exercising an
object nobody is running, and a later test in the same worker reads a value that
disagrees with its own owner, in another file, under some shard splits and not
others, with nothing pointing back at the cause.

The rule, stated once:

    a module that defines a module-level ``__getattr__``, or swaps its own
    ``__class__`` to forward attribute writes, MUST NOT hold a mapping whose
    values are module objects. It keeps the owner's dotted NAME and resolves it
    per use with ``importlib.import_module``, which answers from ``sys.modules``
    and waits on that module's import lock while its body runs.

These tests find the modules the rule binds BY THEIR SHAPE, read off the source
tree, rather than from a list of names. A module that begins mirroring a surface is
therefore covered on the commit that introduces it, with no edit here. A list would
be a second thing to remember, which is the same failure the rule is about.

Two guards keep that generation honest, because a case list derived from a detector
goes SILENT rather than red when the detector stops matching:
``test_the_shape_detector_answers_both_ways`` pins the detector on synthetic sources
it must accept and reject, and ``test_the_discovery_rule_answers_both_ways`` does the
same for the attribute discovery that the purge cases rest on.

The purge cases run in a child process. Importing a module binds it onto its parent
package, so a purge and reimport changes the ``sys.modules`` entry, the parent's
attribute, and any memo the mirror keeps; restoring one of the three in a ``finally``
leaves exactly the split-brain residue this rule forbids, and restoring all three is
a maintenance contract inside a test. A child process has no contract: its residue
leaves with it.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import source_corpus
from mirrored_owner_probe import SENTINEL, one_reexported_pair

from kiro_crew.subprocess_utf8 import UTF8_TEXT

#: Text that must appear for a file to be worth parsing, which is what keeps this
#: scan off the two thirds of the tree that mirror nothing.
_SHAPE_NEEDLES = ("__getattr__", "__class__")


def mirrors_a_surface(tree: ast.Module) -> bool:
    """True when the module resolves or forwards attributes on another's behalf.

    Two spellings do that, and both are visible at module level: a ``__getattr__``
    function, which Python calls for a name the module does not hold, and an
    assignment to some object's ``__class__``, which is how a module installs a
    ``ModuleType`` subclass over itself to intercept writes.
    """
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "__class__":
                    return True
    return False


def owner_module_tables(tree: ast.Module) -> list[str]:
    """Return the names bound to a mapping whose VALUES are module objects.

    Two spellings reach that too: an annotation declaring ``dict[..., ModuleType]``,
    and a dict comprehension whose value expression is a bare name -- a name bound
    to an imported module, since a comprehension over modules is how such a table
    gets built. A mapping of dotted names is neither, because its values are strings
    and its annotation says so.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            targets: list[ast.expr] = [node.target]
            annotation = ast.unparse(node.annotation)
            value: ast.expr | None = node.value
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
            annotation = ""
            value = node.value
        else:
            continue
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if not names:
            continue
        declares_modules = annotation.startswith("dict") and "ModuleType" in annotation
        builds_modules = isinstance(value, ast.DictComp) and isinstance(value.value, ast.Name)
        if declares_modules or builds_modules:
            found.extend(names)
    return sorted(set(found))


def _module_name(path: Path) -> str:
    """Return the dotted import name of the package file at *path*."""
    root = source_corpus.src_root()
    parts = [root.name, *path.relative_to(root).with_suffix("").parts]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _mirroring_modules() -> list[tuple[str, str]]:
    """Return ``(dotted name, source text)`` for every module that mirrors a surface.

    Read through ``source_corpus``, which holds one cached read of the package and
    hands out one parsed tree at a time, so this costs no second walk of the tree and
    retains no trees. Its ``src_root`` is also the importable package, which is what
    keeps these names in step with the runtime half below.
    """
    rows: list[tuple[str, str]] = []
    for path, text, tree in source_corpus.parsed_candidates(require_any=_SHAPE_NEEDLES):
        if mirrors_a_surface(tree):
            rows.append((_module_name(path), text))
    return rows


MIRRORING = _mirroring_modules()

#: One case per mirroring module, so a failure names the module in its own id.
MIRRORING_IDS = [name for name, _text in MIRRORING]


_DETECTOR_CASES: tuple[tuple[str, bool, str], ...] = (
    ("def __getattr__(name):\n    return 1\n", True, "a module-level __getattr__"),
    (
        "import sys\nsys.modules[__name__].__class__ = X\n",
        True,
        "installing a ModuleType subclass over itself",
    ),
    ("class C:\n    def __getattr__(self):\n        return 1\n", False, "a CLASS __getattr__"),
    ("X = 1\n", False, "an ordinary module"),
)

_TABLE_CASES: tuple[tuple[str, list[str], str], ...] = (
    ("from types import ModuleType\n_O: dict[str, ModuleType] = {}\n", ["_O"], "declared"),
    ("_O = {n: mod for mod in MODS for n in mod.__all__}\n", ["_O"], "a comprehension of modules"),
    ("_O = {n: mod.__name__ for mod in MODS for n in mod.__all__}\n", [], "dotted names"),
    ("_O: dict[str, str] = {}\n", [], "a declared name table"),
)


def test_the_shape_detector_answers_both_ways() -> None:
    """The detector the cases below are generated from, pinned on synthetic sources.

    A detector matching nothing would make every parametrized case vanish rather
    than fail, so it is measured directly, on inputs it must accept and inputs it
    must reject.
    """
    for source, expected, why in _DETECTOR_CASES:
        assert mirrors_a_surface(ast.parse(source)) is expected, why
    for source, expected_tables, why in _TABLE_CASES:
        assert owner_module_tables(ast.parse(source)) == expected_tables, why


def test_the_needle_prefilter_admits_every_shape_the_detector_accepts() -> None:
    """The text prefilter must not hide a module the detector would have matched."""
    for source, expected, why in _DETECTOR_CASES:
        if expected:
            assert any(needle in source for needle in _SHAPE_NEEDLES), (
                f"the prefilter would skip {why}, so such a module would never reach "
                "the detector and its absence would read as compliance"
            )


def test_the_discovery_rule_answers_both_ways() -> None:
    """The discovery the purge cases rest on, pinned like the shape detector.

    A table spelling this rule does not recognise turns the purge cases into a skip
    per module rather than a failure, so the rule is measured on modules built here:
    each value spelling it must accept, and the two kinds of value it must pass over.
    """
    probe = ModuleType("kiro_crew_probe_owner")
    probe.__name__ = "kiro_crew.probe_mirror"
    leaf = importlib.import_module("kiro_crew.subprocess_utf8")

    accepted = {
        "a dotted module name": {"UTF8_TEXT": "kiro_crew.subprocess_utf8"},
        "a module object": {"UTF8_TEXT": leaf},
        "an (owner, symbol) pair": {"UTF8_TEXT": ("kiro_crew.subprocess_utf8", "UTF8_TEXT")},
        "a bare name under the parent": {"UTF8_TEXT": "subprocess_utf8"},
    }
    for why, table in accepted.items():
        probe._TABLE = table  # type: ignore[attr-defined]
        assert one_reexported_pair(probe) == (
            "UTF8_TEXT",
            "kiro_crew.subprocess_utf8",
        ), f"discovery missed {why}"

    rejected = {
        "a dunder key": {"__spec__": "kiro_crew.subprocess_utf8"},
        "an attribute the owner lacks": {"no_such_attribute_here": "kiro_crew.subprocess_utf8"},
        "an owner outside this package": {"getcwd": "os"},
        "a value that is not a name": {"UTF8_TEXT": 17},
    }
    for why, table in rejected.items():
        probe._TABLE = table  # type: ignore[attr-defined]
        assert one_reexported_pair(probe) is None, f"discovery accepted {why}"

    probe.__dunder_table__ = {  # type: ignore[attr-defined]
        "UTF8_TEXT": "kiro_crew.subprocess_utf8"
    }
    del probe._TABLE  # type: ignore[attr-defined]
    assert one_reexported_pair(probe) is None, "discovery read a dunder-named table"


def test_the_scan_finds_the_modules_that_mirror_a_surface() -> None:
    """The scan reaches the package at all, so an empty result is a broken scan."""
    assert MIRRORING, f"no mirroring module found under {source_corpus.src_root()}"


@pytest.mark.parametrize(("name", "text"), MIRRORING, ids=MIRRORING_IDS)
def test_no_mirroring_module_stores_a_resolved_owner(name: str, text: str) -> None:
    """The rule, read off the source of every module the scan finds."""
    tables = owner_module_tables(ast.parse(text))
    assert tables == [], (
        f"{name} holds {tables}, a mapping to resolved owner MODULES and so a second "
        "storage location beside sys.modules. Hold the owner's dotted NAME and resolve "
        "it per use with importlib.import_module."
    )


@pytest.mark.parametrize(("name", "text"), MIRRORING, ids=MIRRORING_IDS)
def test_a_mirroring_module_resolves_its_owners_through_the_import_system(
    name: str, text: str
) -> None:
    """The other half of the rule: resolution asks the import system every time."""
    assert "importlib.import_module(" in text, (
        f"{name} mirrors a surface but never calls importlib.import_module, so its "
        "owners are not being resolved from sys.modules"
    )


@pytest.mark.parametrize(("name", "text"), MIRRORING, ids=MIRRORING_IDS)
def test_no_live_mapping_holds_a_module_object(name: str, text: str) -> None:
    """The same rule measured on the imported module, which also catches memoisation.

    A table built from dotted names but memoising the module it resolved satisfies
    the source cases and fails here, so the two are not redundant. This one only
    reads, so it runs in process.
    """
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - importability is another test's subject
        pytest.skip(f"{name} does not import here: {type(exc).__name__}")
    offenders = {
        attr: sorted(key for key, val in value.items() if isinstance(val, ModuleType))
        for attr, value in list(vars(module).items())
        if isinstance(value, dict) and any(isinstance(val, ModuleType) for val in value.values())
    }
    assert offenders == {}, (
        f"{name} holds resolved owner modules at runtime, so a purged owner stays "
        "invisible through it: " + repr({attr: keys[:3] for attr, keys in offenders.items()})
    )


def _parent_attribute_divergences() -> set[str]:
    """Submodules whose parent-package attribute is not the object ``sys.modules`` holds.

    The same one-storage question this rule asks of a mirroring module, asked of the
    interpreter: ``importlib`` finishes a load with ``setattr(parent, child, module)``,
    so a purge and reimport that puts back only the ``sys.modules`` entry leaves the
    parent naming the discarded module. That is a second reference to a value the
    authoritative store has moved past.
    """
    diverged: set[str] = set()
    for name, _text in MIRRORING:
        module = sys.modules.get(name)
        if module is None:
            continue
        prefix = f"{name}."
        for loaded_name, loaded in list(sys.modules.items()):
            if not loaded_name.startswith(prefix):
                continue
            leaf = loaded_name[len(prefix) :]
            if "." in leaf:
                continue
            bound = getattr(module, leaf, None)
            if isinstance(bound, ModuleType) and bound is not loaded:
                diverged.add(f"{name}.{leaf} is not sys.modules[{loaded_name!r}]")
    return diverged


def _sentinel_sightings() -> set[str]:
    """Places holding the probe's sentinel, which only the probe ever writes."""
    seen: set[str] = set()
    for name, _text in MIRRORING:
        for candidate_name, candidate in list(sys.modules.items()):
            if candidate_name != name and not candidate_name.startswith(f"{name}."):
                continue
            for attr, value in list(vars(candidate).items()):
                if isinstance(value, str) and value == SENTINEL:
                    seen.add(f"{candidate_name}.{attr}")
    return seen


def test_the_probe_leaves_this_process_untouched() -> None:
    """The rule turned on this file's own fixture, which is where it is easiest to break.

    A probe that purged in process would have to put back the ``sys.modules`` entry,
    the parent package's attribute and any memo the mirror keeps; missing one leaves
    the split-brain the rule forbids, for whatever test runs next in this worker.
    Running the probe in a child process is what removes that obligation, and this
    case is what proves the obligation is really gone rather than merely intended.

    Baselined rather than absolute: another file's fixture may already have left a
    divergence in this worker, and that is not this probe's doing. Only a divergence
    the probe ADDS fails here. The sentinel needs no baseline, since nothing else
    writes it.
    """
    before = _parent_attribute_divergences()
    verdicts = _probe_one_child()
    assert verdicts, "the probe child returned nothing, so this case measured nothing"

    introduced = _parent_attribute_divergences() - before
    assert introduced == set(), (
        "running the purge probe left this process holding a parent attribute that "
        f"disagrees with sys.modules: {sorted(introduced)}"
    )
    assert _sentinel_sightings() == set(), (
        "the probe's sentinel value is still reachable in this process: "
        f"{sorted(_sentinel_sightings())}"
    )


def _probe_one_child() -> dict[str, dict[str, object]]:
    """Run the probe in a child process once and return its verdicts by module name."""
    if not MIRRORING_IDS:  # pragma: no cover - the empty scan has its own case
        return {}
    test_dir = Path(__file__).resolve().parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(test_dir.parent / "src"), str(test_dir), env.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, str(test_dir / "mirrored_owner_probe.py"), *MIRRORING_IDS],
        capture_output=True,
        check=True,
        env=env,
        **UTF8_TEXT,
    )
    verdicts = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    return {str(verdict["module"]): verdict for verdict in verdicts}


def _probe_verdicts() -> dict[str, dict[str, object]]:
    """Run every module's purge probe in ONE child process and return its verdicts.

    One child rather than one per module, because the probe's cost is the import of
    the package under test. Nothing is restored afterwards: the residue of a purge
    and reimport -- the ``sys.modules`` entry, the parent package's attribute, any
    memo the mirror keeps -- exits with the child.
    """
    return _probe_one_child()


PROBE_VERDICTS = None


def _verdict(name: str) -> dict[str, object]:
    global PROBE_VERDICTS
    if PROBE_VERDICTS is None:
        PROBE_VERDICTS = _probe_verdicts()
    verdict = PROBE_VERDICTS.get(name)
    if verdict is None:
        pytest.fail(f"the probe child returned no verdict for {name}")
    return verdict


@pytest.mark.parametrize(("name", "text"), MIRRORING, ids=MIRRORING_IDS)
def test_a_read_through_the_mirror_resolves_the_owner_in_sys_modules(name: str, text: str) -> None:
    """A read answers from the module ``sys.modules`` holds, not from an earlier one.

    Resolving only the OWNER through the import system while the VALUE stays bound
    in the mirroring module's own namespace is not half of this rule -- it is its own
    defect. A read then returns the binding made before a purge while a write
    resolves the module ``sys.modules`` now holds, and ``monkeypatch`` restores by
    reassignment: it reads the attribute to remember it, then assigns the remembered
    value back. Teardown therefore installs a pre-purge value into the fresh module,
    for the life of the worker.
    """
    verdict = _verdict(name)
    if "skipped" in verdict:
        pytest.skip(f"{name}: {verdict['skipped']}")
    assert verdict.get("reimport_is_new") is True, (
        f"{name}: the probe's reimport returned the same object, so it measured "
        "nothing -- there was no purge to see through"
    )
    assert verdict.get("read_resolves") is True, (
        f"reading {name}.{verdict.get('attribute')} did not answer from the "
        f"{verdict.get('owner')} that sys.modules holds (got {verdict.get('read')!r}), so a "
        "read and a write through this module name different objects and a patch "
        "fixture's teardown writes a pre-purge value into the fresh module"
    )


@pytest.mark.parametrize(("name", "text"), MIRRORING, ids=MIRRORING_IDS)
def test_a_purged_owner_is_not_retained_by_the_mirroring_module(name: str, text: str) -> None:
    """No mapping keeps the discarded owner once it has been replaced."""
    verdict = _verdict(name)
    if "skipped" in verdict:
        pytest.skip(f"{name}: {verdict['skipped']}")
    assert verdict.get("retained") == [], (
        f"{name} keeps the discarded {verdict.get('owner')} in "
        f"{verdict.get('retained')} after a purge and reimport, so reads and writes "
        "through it reach a module nothing else sees"
    )
