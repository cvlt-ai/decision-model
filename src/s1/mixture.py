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

import hashlib
import json
import random
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
SYNTHETIC = {"bespoke-synthetic"}  # Nimble c2d pairs (GPT-5.6, model-checked, no stated license)


def render(state, instructions, options: list[tuple[str, str]]) -> str:
    """Shared prompt shape for all primitives. options = [(key, desc), ...]

    A description that is just the key reformatted (underscores<->spaces, e.g.
    banking77 `why_verify_identity` -> `why verify identity`) adds no
    information, so it is dropped to `[i] key`. This keeps 77-option banking77
    prompts ~520 tokens instead of ~1037, so the STATE survives the 1024 budget.
    """
    st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    lines = [f"State:\n{st}", "", f"Question: {instructions}", "", "Options:"]
    for i, (k, desc) in enumerate(options, 1):
        lines.append(f"[{i}] {_option_label(k, desc)}")
    lines += ["", "Answer:"]
    return "\n".join(lines)


def _option_label(key: str, desc: str) -> str:
    k = str(key).strip()
    d = str(desc).strip()
    if d and d.replace(" ", "_").lower() != k.replace(" ", "_").lower():
        return f"{k}: {d}"
    return k


def _negate_q(instructions: str) -> str:
    return f"Is it NOT the case that the following is true? {instructions}"


def _shuffled_options(uid: str, options: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Deterministic permutation of an option list, seeded by uid.

    Same uid -> same order (reproducible build, train/val never disagree).
    Different rows -> different orders, so the model sees P(key | option-SET),
    not P(key | option-LIST-ORDER). The keys travel with the description, so the
    gold target (a key) is unchanged by the reordering.
    """
    seed = int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "big")
    opts = list(options)
    random.Random(seed).shuffle(opts)
    return opts


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


def build_banking77(cap: int | None = None, order_aug: bool = False) -> list[dict]:
    ds = load_from_disk(str(RAW / "banking77__train"))
    labels = sorted({str(r["label_text"]) for r in ds})
    if len(labels) != 77:
        raise ValueError(f"banking77 train lost labels: {len(labels)}")
    criteria = {lab: lab.replace("_", " ") for lab in labels}
    rows = []
    for i, r in enumerate(ds):
        if cap and i >= cap:
            break
        uid = f"banking77/{i}"
        gold = str(r["label_text"])
        rows.append(_row(uid, "banking77",
                         render(str(r["text"]), B77_INSTR, list(criteria.items())),
                         gold, "mit"))
        if order_aug:
            rows.append(_row(uid + "/s", "banking77",
                             render(str(r["text"]), B77_INSTR,
                                    _shuffled_options(uid, list(criteria.items()))),
                             gold, "mit"))
    return rows


def build_go_emotions(cap: int | None = None, order_aug: bool = False) -> list[dict]:
    ds = load_from_disk(str(RAW / "go_emotions__train"))
    criteria = [(lab, lab) for lab in GOE_LABELS]
    rows = []
    for i, r in enumerate(ds):
        if cap and i >= cap:
            break
        uid = f"go_emotions/{i}"
        gold = GOE_LABELS[int(r["labels"][0])]
        rows.append(_row(uid, "go_emotions",
                         render(str(r["text"]), GOE_INSTR, criteria), gold, "apache-2.0"))
        if order_aug:
            rows.append(_row(uid + "/s", "go_emotions",
                             render(str(r["text"]), GOE_INSTR,
                                    _shuffled_options(uid, criteria)),
                             gold, "apache-2.0"))
    return rows


def build_injection(cap: int | None = None, order_aug: bool = False) -> list[dict]:
    # noul = 2 options; order_aug accepted but unused (2-option swap is the
    # negation twin's job, already emitted by _noul_pairs)
    ds = load_from_disk(str(RAW / "injection__train"))
    rows = []
    for i, r in enumerate(ds):
        if cap and i >= cap:
            break
        rows += _noul_pairs(f"injection/{i}", "injection", str(r["text"]), INJ_INSTR,
                            INJ_CRIT["true"], INJ_CRIT["false"],
                            int(r["label"]) == 1, "apache-2.0")
    return rows


def build_severity(cap: int | None = None, order_aug: bool = False) -> list[dict]:
    ds = load_from_disk(str(RAW / "severity__train"))
    opts = [(str(i), lev) for i, lev in enumerate(SEV_LEVELS)]  # 0-based wire levels
    rows = []
    for i, r in enumerate(ds):
        if cap and i >= cap:
            break
        uid = f"severity/{i}"
        sev = str(r["severity"]).strip().lower()
        if sev not in SEV_LEVELS:
            continue
        rows.append(_row(uid, "severity",
                         render(str(r["function"]), SEV_INSTR, opts),
                         str(SEV_LEVELS.index(sev)), "mit"))
        if order_aug:
            rows.append(_row(uid + "/s", "severity",
                             render(str(r["function"]), SEV_INSTR,
                                    _shuffled_options(uid, opts)),
                             str(SEV_LEVELS.index(sev)), "mit"))
    return rows


def build_boolq(cap: int | None = None, order_aug: bool = False) -> list[dict]:
    """cc-by-sa: only with include_sa (research-only release line). noul = 2 options."""
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


def _nimble_state_to_str(state) -> str:
    """Nimble `input.state` is either a list of {speaker, text} turns, a list
    of plain statement strings, a single string, or a dict. `render` wants a
    plain string. Turns -> `Speaker: text` lines; plain strings -> one line
    each (keeps the multi-part structure the c2d curation relies on)."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return json.dumps(state, ensure_ascii=False)
    lines = []
    for turn in state:
        if isinstance(turn, dict):
            spk = str(turn.get("speaker", "")).strip()
            txt = str(turn.get("text", "")).strip()
            lines.append(f"{spk}: {txt}" if spk else txt)
        else:
            lines.append(str(turn).strip())
    return "\n".join(lines)


def build_nimble_c2d(cap: int | None = None, order_aug: bool = False) -> list[dict]:
    """Nimble contrastive (c2d) pairs -> Jev decision rows (bespoke-synthetic).

    Their `input` is the Jev wire schema: state + one questions.<field> with
    {type, criteria, instructions}. We map all three types onto our render():
      choice -> criteria dict  -> options [(key, desc)],  target = reference key
      noul   -> criteria dict  -> options true/false,      target = "true"/"false"
      score  -> criteria LIST  -> options [(str(i), desc)], target = str(i)
    The score mapping is exactly our severity 0-based wire convention.
    Both the base and counterfactual rows are kept (the flip IS the signal);
    order_aug emits a shuffled-option twin for the choice/score rows.
    License `bespoke-synthetic` gates it behind --include-c2d.
    """
    src = RAW / "nimble_c2d_train.jsonl"
    rows: list[dict] = []
    for i, line in enumerate(src.read_text().splitlines()):
        if not line.strip():
            continue
        if cap and i >= cap:
            break
        rec = json.loads(line)
        state = _nimble_state_to_str(rec["input"].get("state"))
        q = next(iter(rec["input"]["questions"].values()))
        qtype = q.get("type")
        instr = str(q.get("instructions", "")).strip()
        target = rec["reference"]["target"]
        uid = f"nimble/{rec['id']}"
        if qtype == "choice":
            options = list(q.get("criteria", {}).items())
            tgt = str(target)
            rows.append(_row(uid, "nimble", render(state, instr, options), tgt,
                             "bespoke-synthetic"))
            if order_aug:
                rows.append(_row(uid + "/s", "nimble",
                                 render(state, instr, _shuffled_options(uid, options)),
                                 tgt, "bespoke-synthetic"))
        elif qtype == "noul":
            crit = q.get("criteria", {})
            options = [("true", crit.get("true", "the proposition holds")),
                       ("false", crit.get("false", "the proposition does not hold"))]
            tgt = "true" if target in (True, "true") else "false"
            rows.append(_row(uid, "nimble", render(state, instr, options), tgt,
                             "bespoke-synthetic"))
            # noul = 2 options; the flip twin is already a distinct c2d row,
            # so order_aug (swap) is redundant here -> accept but ignore
        elif qtype == "score":
            crit = q.get("criteria", [])
            options = [(str(idx), str(desc)) for idx, desc in enumerate(crit)]
            tgt = str(int(target))
            rows.append(_row(uid, "nimble", render(state, instr, options), tgt,
                             "bespoke-synthetic"))
            if order_aug:
                rows.append(_row(uid + "/s", "nimble",
                                 render(state, instr, _shuffled_options(uid, options)),
                                 tgt, "bespoke-synthetic"))
        else:
            raise ValueError(f"nimble row {rec['id']}: unknown question type {qtype}")
    return rows


BUILDERS = {
    "banking77": build_banking77,
    "go_emotions": build_go_emotions,
    "injection": build_injection,
    "severity": build_severity,
    "boolq": build_boolq,
    "nimble": build_nimble_c2d,
}
DEFAULT_FAMILIES = ["banking77", "go_emotions", "injection", "severity"]  # permissive only


def build_mixture(families: list[str] | None = None, cap: int | None = None,
                  include_sa: bool = False, order_aug: bool = False,
                  include_c2d: bool = False) -> list[dict]:
    fams = families or (DEFAULT_FAMILIES
                        + (["boolq"] if include_sa else [])
                        + (["nimble"] if include_c2d else []))
    rows: list[dict] = []
    for f in fams:
        for row in BUILDERS[f](cap, order_aug):
            lic_ok = (row["license"] in PERMISSIVE
                      or (row["license"] in SA and include_sa)
                      or (row["license"] in SYNTHETIC and include_c2d))
            if not lic_ok:
                raise ValueError(
                    f"{row['uid']} is {row['license']}; not enabled "
                    f"(permissive only, or use --include-sa / --include-c2d)")
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
    ap.add_argument("--order-aug", action="store_true",
                    help="emit a shuffled-option twin for each choice row (order invariance)")
    ap.add_argument("--include-c2d", action="store_true",
                    help="add Nimble c2d contrastive pairs (bespoke-synthetic)")
    ap.add_argument("--families", default=None)
    a = ap.parse_args()
    fams = a.families.split(",") if a.families else None
    print(json.dumps(write_mixture(Path(a.out), families=fams, cap=a.cap,
                                   include_sa=a.include_sa, order_aug=a.order_aug,
                                   include_c2d=a.include_c2d), indent=1))
