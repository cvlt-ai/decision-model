"""Nimble c2d contrastive pairs -> Jev decision rows.

Covers the three question types (choice / noul / score) against the real
third-party data (pinned copy at data/raw/nimble_c2d_train.jsonl), the
flip-signal property, the license gate, and order-aug invariance.
"""
import json
from pathlib import Path

import pytest

import sys
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from s1.mixture import build_mixture, build_nimble_c2d, _nimble_state_to_str  # noqa: E402

RAW = ROOT / "data" / "raw" / "nimble_c2d_train.jsonl"


def _target_in_prompt(prompt: str, target: str) -> bool:
    """Gold key must appear in the Options block (readout needs the token)."""
    opts = prompt.split("Options:\n", 1)
    assert len(opts) == 2, "no Options block"
    return target in opts[1]


@pytest.mark.skipif(not RAW.exists(), reason="nimble c2d data not present")
class TestNimbleC2D:
    def test_row_shape_and_family(self):
        rows = build_nimble_c2d(cap=20)
        assert rows, "no rows produced"
        for r in rows:
            assert set(r) == {"uid", "family", "prompt", "target", "license"}
            assert r["family"] == "nimble"
            assert r["license"] == "bespoke-synthetic"
            assert r["uid"].startswith("nimble/")
            assert r["prompt"].startswith("State:\n")
            assert "Question:" in r["prompt"] and "Options:" in r["prompt"]
            assert r["prompt"].rstrip().endswith("Answer:")

    def test_all_three_types_present_and_target_valid(self):
        rows = build_nimble_c2d()  # full set
        assert len(rows) >= 2676  # base + counterfactual, one row per record
        # every target must be one of the offered option keys
        for r in rows:
            opts = r["prompt"].split("Options:\n", 1)[1].split("Answer:", 1)[0]
            # target token present in the option listing
            assert r["target"] in opts, f"target {r['target']!r} not in options"

    def test_state_turns_rendered_with_speaker(self):
        rec = json.loads(RAW.read_text().splitlines()[0])
        st = rec["input"]["state"]
        rendered = _nimble_state_to_str(st)
        # multi-turn list -> 'Speaker: text' lines, one per turn
        assert rendered.count("\n") == len(st) - 1
        first = st[0]
        assert f"{first['speaker']}: {first['text']}" in rendered

    def test_flip_signal_preserved(self):
        # Each family is one contrast pair: base + counterfactual, same family.
        # The c2d method guarantees the target flips between them.
        recs = [json.loads(l) for l in RAW.read_text().splitlines() if l.strip()]
        from collections import defaultdict
        fam = defaultdict(dict)
        for rec in recs:
            fam[rec["family"]][rec["variant"]] = rec
        pairs = {k: d for k, d in fam.items() if "base" in d and "counterfactual" in d}
        flips = sum(1 for d in pairs.values()
                    if d["base"]["reference"]["target"] !=
                    d["counterfactual"]["reference"]["target"])
        assert len(pairs) == 1338, f"expected 1338 contrast pairs, got {len(pairs)}"
        assert flips == len(pairs), f"only {flips}/{len(pairs)} pairs flip the target"

    def test_license_gate(self):
        # without include_c2d the default mix must NOT contain nimble rows
        mix = build_mixture(include_sa=True, order_aug=True)
        assert all(r["family"] != "nimble" for r in mix)
        # with include_c2d it does
        mix2 = build_mixture(include_sa=True, order_aug=True, include_c2d=True)
        assert any(r["family"] == "nimble" for r in mix2)

    def test_order_aug_invariance_choice(self):
        # choice/score rows get a /s twin; same target, options reordered (set-equal)
        rows = build_nimble_c2d(order_aug=True)
        base_by_uid = {r["uid"]: r for r in rows if not r["uid"].endswith("/s")}
        twins = [r for r in rows if r["uid"].endswith("/s")]
        assert twins, "no order-aug twins"
        for t in twins:
            b = base_by_uid[t["uid"][:-2]]
            assert b["target"] == t["target"]
            # option contents identical as a set (index prefix may change)
            def optset(p):
                body = p.split("Options:\n", 1)[1].split("Answer:", 1)[0]
                items = []
                for ln in body.splitlines():
                    if ln.startswith("[") and "] " in ln:
                        items.append(ln.split("] ", 1)[1])
                return set(items)
            assert optset(b["prompt"]) == optset(t["prompt"])
