"""Training mixture builder: data/raw/*__train -> Jev decision rows.

Contract:
  * prompt wording reuses the EXACT INSTRUCTIONS/criteria constants from the eval
    adapters - if training text differs from eval text, eval gains are meaningless
  * one row = {"uid","family","prompt","target","license"}
  * every noul row also gets a negated twin (flipped question + flipped gold):
    negation consistency is our headline gate (Laya 0.710 violation) and the only
    way to train it without violating P(x)+P(-x)=1 at data level is to SHOW it
  * license gate: default permissive-only (MIT/Apache); --include-sa adds cc-by-sa
    (boolq) which forces the research-only release line
  * deterministic: sorted iteration, fixed caps, no RNG

Reads pinned local copies (data/raw), never the network; never touches data/holdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
RAW = ROOT / "data" / "raw"

# same constants the eval adapters use (importing, not retyping, is the point)
from eval.datasets.banking77 import INSTRUCTIONS as B77_INSTR  # noqa: E402
from eval.datasets.go_emotions import INSTRUCTIONS as GOE_INSTR  # noqa: E402
from eval.datasets.go_emotions import LABELS as GOE_LABELS  # noqa: E402
from eval.datasets.injection import CRIT as INJ_CRIT  # noqa: E402
from eval.datasets.injection import INSTR as INJ_INSTR  # noqa: E402
from eval.datasets.severity import INSTRUCTIONS as SEV_INSTR  # noqa: E402
from eval.datasets.severity import LEVELS as SEV_LEVELS  # noqa: E402

PERMISSIVE = {"mit", "apache-2.0"}
SA = {"cc-by-sa-3.0", "cc-by-sa-4.0"}


def render(state, instructions, options: list[tuple[str, str]]) -> str:
    """Shared prompt shape for all primitives. options = [(key, desc), ...]"""
    st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    lines = [f"State:\n{st}", "", f"Question: {instructions}", "", "Options:"]
    for i, (k, desc) in enumerate(options, 1):
        lines.append(f"[{i}] {k}: {desc}")
    lines += ["", "Answer:"]
    return "\n".join(lines)


def _negate_q(instructions: str) -> str:
    return f"Is it NOT the case that the following is true? {instructions}"


def _row(uid, family, prompt, target, license_):
    return {"uid": uid, "family": family, "prompt": prompt, "target": target,
            "license": license_}


def _noul_pairs(uid, family, state, instructions, crit_true, crit_false, gold_bool, lic):
    """One noul question + its negated twin (gold flipped too)."""
    opts = [("true", crit_true), ("false", crit_false)]
    rows = [_row(f"{uid}/p", family, render(state, instructions, opts),
                 "true" if gold_bool else "false", lic)]
    rows.append(_row(f"{uid}/n", family, render(state, _negate_q(instructions),
                     [("true", "the original proposition does NOT hold"),
                      ("false", "the original proposition holds")]),
                 "false" if gold_bool else "true", lic))
    return rows


def build_banking77(cap: int | None = None) -> list[dict]:
    ds = load_from_disk(str(RAW / "banking77__train"))
    labels = sorted({str(r["label_text"]) for r in ds})
    if len(labels) != 77:
        raise ValueError(f"banking77 train lost labels: {len(labels)}")
    criteria = {lab: lab.replace("_", " ") for lab in labels}
    rows = []
    for i, r in enumerate(ds):
        if cap and i >= cap:
            break
        gold = str(r["label_text"])
        rows.append(_row(f"banking77/{i}", "banking77",
                         render(str(r["text"]), B77_INSTR, list(criteria.items())),
                         gold, "mit"))
    return rows


def build_go_emotions(cap: int | None = None) -> list[dict]:
    ds = load_from_disk(str(RAW / "go_emotions__train"))
    criteria = [(lab, lab) for lab in GOE_LABELS]
    rows = []
    for i, r in enumerate(ds):
        if cap and i >= cap:
            break
        gold = GOE_LABELS[int(r["labels"][0])]
        rows.append(_row(f"go_emotions/{i}", "go_emotions",
                         render(str(r["text"]), GOE_INSTR, criteria), gold, "apache-2.0"))
    return rows


def build_injection(cap: int | None = None) -> list[dict]:
    ds = load_from_disk(str(RAW / "injection__train"))
    rows = []
    for i, r in enumerate(ds):
        if cap and i >= cap:
            break
        rows += _noul_pairs(f"injection/{i}", "injection", str(r["text"]), INJ_INSTR,
                            INJ_CRIT["true"], INJ_CRIT["false"],
                            int(r["label"]) == 1, "apache-2.0")
    return rows


def build_severity(cap: int | None = None) -> list[dict]:
    ds = load_from_disk(str(RAW / "severity__train"))
    opts = [(str(i), lev) for i, lev in enumerate(SEV_LEVELS)]  # 0-based wire levels
    rows = []
    for i, r in enumerate(ds):
        if cap and i >= cap:
            break
        sev = str(r["severity"]).strip().lower()
        if sev not in SEV_LEVELS:
            continue
        rows.append(_row(f"severity/{i}", "severity",
                         render(str(r["function"]), SEV_INSTR, opts),
                         str(SEV_LEVELS.index(sev)), "mit"))
    return rows


def build_boolq(cap: int | None = None) -> list[dict]:
    """cc-by-sa: only with include_sa (research-only release line)."""
    ds = load_from_disk(str(RAW / "boolq__train"))
    rows = []
    for i, r in enumerate(ds):
        if cap and i >= cap:
            break
        q = str(r["question"])
        instr = (f"Based only on the passage in the state, is the answer to the "
                 f"question '{q}' yes?")
        rows += _noul_pairs(f"boolq/{i}", "boolq",
                            {"passage": r["passage"], "question": q}, instr,
                            "the passage supports a yes answer",
                            "the passage supports a no answer",
                            bool(r["answer"]), "cc-by-sa-3.0")
    return rows


BUILDERS = {
    "banking77": build_banking77,
    "go_emotions": build_go_emotions,
    "injection": build_injection,
    "severity": build_severity,
    "boolq": build_boolq,
}
DEFAULT_FAMILIES = ["banking77", "go_emotions", "injection", "severity"]  # permissive only


def build_mixture(families: list[str] | None = None, cap: int | None = None,
                  include_sa: bool = False) -> list[dict]:
    fams = families or (DEFAULT_FAMILIES + (["boolq"] if include_sa else []))
    rows: list[dict] = []
    for f in fams:
        for row in BUILDERS[f](cap):
            if row["license"] not in PERMISSIVE and not include_sa:
                raise ValueError(
                    f"{row['uid']} is {row['license']}; permissive mixture "
                    f"requires include_sa=True (research-only line)")
            rows.append(row)
    uids = [r["uid"] for r in rows]
    assert len(uids) == len(set(uids)), "duplicate uids in mixture"
    return rows


def write_mixture(out: Path, **kw) -> dict:
    rows = build_mixture(**kw)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    fams: dict[str, int] = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    return {"n": len(rows), "families": fams, "path": str(out)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "mixture_v1.jsonl"))
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--include-sa", action="store_true")
    ap.add_argument("--families", default=None)
    a = ap.parse_args()
    fams = a.families.split(",") if a.families else None
    print(json.dumps(write_mixture(Path(a.out), families=fams, cap=a.cap,
                                   include_sa=a.include_sa), indent=1))
