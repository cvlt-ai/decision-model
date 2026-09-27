"""tsi builder: TaskSource general-instruction rows -> Jev choice rows.

Covers the render (bare-key options, state carries the content), the gold-in-options
invariant, the U+2028 split guard, order-aug invariance, and the license gate.
Runs on the pinned data/raw/tsi_train.jsonl (skips cleanly if absent).
"""
import json
from pathlib import Path

import pytest

import sys
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from s1.mixture import build_mixture, build_tsi  # noqa: E402

RAW = ROOT / "data" / "raw" / "tsi_train.jsonl"


def _optset(p):
    body = p.split("Options:\n", 1)[1].split("Answer:", 1)[0]
    return set(ln.split("] ", 1)[1] for ln in body.splitlines()
               if ln.startswith("[") and "] " in ln)


@pytest.mark.skipif(not RAW.exists(), reason="tsi data not present")
class TestTsi:
    def test_row_shape_and_family(self):
        rows = build_tsi(cap=20)
        assert rows
        for r in rows:
            assert set(r) == {"uid", "family", "prompt", "target", "license"}
            assert r["family"] == "tsi"
            assert r["license"] == "tsi-perm"
            assert r["uid"].startswith("tsi/")
            assert r["prompt"].startswith("State:\n")
            assert "Question:" in r["prompt"] and "Options:" in r["prompt"]
            assert r["prompt"].rstrip().endswith("Answer:")

    def test_gold_always_among_options(self):
        rows = build_tsi()
        for r in rows:
            assert r["target"] in r["prompt"].split("Options:\n", 1)[1]

    def test_all_source_rows_valid_json_after_split_guard(self):
        # the whole point of the split("\n") + skip guard: every parseable line yields
        # a row, and a stray bad line never crashes the build
        n_src = sum(1 for ln in RAW.read_text().split("\n") if ln.strip())
        rows = build_tsi()
        assert len(rows) >= n_src  # >= because order_aug off here, one row per line
        assert all(r["target"] for r in rows)

    def test_license_gate(self):
        mix = build_mixture(include_sa=True, order_aug=True, include_c2d=True)
        assert all(r["family"] != "tsi" for r in mix)
        mix2 = build_mixture(include_sa=True, order_aug=True, include_c2d=True,
                             include_tsi=True)
        assert any(r["family"] == "tsi" for r in mix2)

    def test_order_aug_invariance(self):
        rows = build_tsi(cap=30, order_aug=True)
        base = {r["uid"]: r for r in rows if not r["uid"].endswith("/s")}
        twins = [r for r in rows if r["uid"].endswith("/s")]
        assert twins, "no order-aug twins"
        for t in twins:
            b = base[t["uid"][:-2]]
            assert b["target"] == t["target"]
            assert _optset(b["prompt"]) == _optset(t["prompt"])

    def test_short_rows_no_context_blowup(self):
        # TSI is short (max ~625 tokens measured); the builder must not emit
        # anything absurdly long that would OOM a 4096 run
        rows = build_tsi(cap=200)
        assert all(len(r["prompt"]) < 8000 for r in rows)
