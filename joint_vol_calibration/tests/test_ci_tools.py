"""
test_ci_tools.py: the developer and CI tools under scripts/ci and scripts/bench
that are not part of the daily bot's gates (those are in test_infra_config.py).

  * mutation_check.py (`make mutation`): every curated mutant still names a
    site that occurs exactly once, so a refactor that copies or rewrites a
    mutated line fails here instead of turning the weekly run red as "stale".
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)             # dataclasses look the module up
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mutation(monkeypatch):
    return _load("scripts/ci/mutation_check.py", "mutation_check_under_test", monkeypatch)


class TestMutationCheck:
    def test_every_mutant_site_occurs_exactly_once(self, mutation):
        """ENG-2: C23's analytic Jacobian copied `beta = kappa - 1j * rho * sig * z`
        into _heston_log_cf_grad, the rho mutant matched twice, `make mutation`
        reported it 'stale' and exited 1. Each site now carries the next line."""
        assert mutation.stale_sites() == {}
        for name, rel, old, new in mutation.MUTANTS:
            assert old != new, name
            assert (ROOT / rel).read_text().count(new) == 0, f"{name}: the mutated text is already there"

    def test_both_copies_of_the_cf_line_are_mutated(self, mutation):
        src = (ROOT / mutation.HESTON).read_text()
        line = "beta = kappa - 1j * rho * sig * z"
        sites = [old for name, rel, old, _ in mutation.MUTANTS if line in old]
        assert src.count(line) == len(sites) == 2
        assert len({src.index(s) for s in sites}) == 2               # two different places

    def test_mutant_tests_exist_and_name_known_mutants(self, mutation):
        names = [m[0] for m in mutation.MUTANTS]
        assert len(names) == len(set(names))
        assert set(mutation.MUTANT_TESTS) <= set(names)
        assert set(mutation.EQUIVALENT) <= set(names)
        for rel in mutation.all_tests():
            assert (ROOT / rel).is_file(), rel

    def test_a_duplicated_site_is_reported_stale(self, mutation, tmp_path):
        name, rel, old, _ = mutation.MUTANTS[0]
        for _, r, *_ in mutation.MUTANTS:
            (tmp_path / r).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / r).write_text((ROOT / r).read_text())
        (tmp_path / rel).write_text((ROOT / rel).read_text() + "\n# " + old + "\n")
        assert mutation.stale_sites(tmp_path) == {name: 2}

