"""S1 SFT training (M2 line): LoRA on a causal LM over the decision mixture.

Recipe follows the Open-Jev validation point (arXiv 2609.23959): small causal LM +
LoRA, decision made later by logit readout over answer-key tokens (readout lives in
the server, not here). This script is the SFT stage only: next-token CE masked to
the target span. RLCD / temperature scaling come after this checkpoint exists.

    # real run (GPUs free):
    cd /home/user/decision-model
    .venv/bin/python scripts/train_s1.py --model Qwen/Qwen3.5-4B --epochs 2

    # smoke (CPU, tiny model, 2 steps) to prove the loop end-to-end:
    .venv/bin/python scripts/train_s1.py --model Qwen/Qwen2.5-0.5B --device cpu \
        --max-len 256 --steps 2 --out checkpoints/smoke

Refuses to start on GPU unless >=8 GB free (llama-server courtesy; --force overrides).
Built-in cheap decontamination: any mixture row whose rendered prompt contains a
verbatim holdout state string is DROPPED and counted (exact-substring pass; MinHash
fuzzy pass is scripts/decontaminate.py, still TODO before any release).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# `python scripts/train_s1.py` puts scripts/ on sys.path, not the repo root, so
# `import eval...` (and s1 when not pip-installed) would fail. Self-bootstrap:
# works from any cwd, with or without PYTHONPATH, venv-agnostic.
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torch.utils.data import Dataset
MIXTURE = ROOT / "data" / "processed" / "mixture_v1.jsonl"
HOLDOUT = ROOT / "data" / "holdout"


def gpu_free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / 2**30


class DecisionSFT(Dataset):
    """Tokenizes prompt+target; labels are -100 on the prompt span."""

    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
        t = self.tok(r["target"] + self.tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p + t)[-self.max_len :]
        # target keeps its length; prompt truncates from the left if over budget
        n_prompt_kept = max(0, len(ids) - len(t))
        labels = [-100] * n_prompt_kept + ids[n_prompt_kept:]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    out = {
        "input_ids": torch.tensor([b["input_ids"] + [pad_id] * (L - len(b["input_ids"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (L - len(b["input_ids"])) for b in batch]),
        "labels": torch.tensor([[-100] * (L - len(b["labels"])) + b["labels"] for b in batch]),
    }
    return out


def load_rows(mixture: Path, decontam: bool):
    rows = [json.loads(ln) for ln in mixture.read_text().splitlines()]
    stats = {"n_raw": len(rows)}
    if decontam and HOLDOUT.exists():
        import pyarrow.parquet as pq

        from eval.datasets.banking77 import INSTRUCTIONS as B77_INSTR
        from eval.datasets.go_emotions import INSTRUCTIONS as GOE_INSTR
        from eval.datasets.injection import INSTR as INJ_INSTR
        from eval.datasets.pubhealth import INSTRUCTIONS as PUB_INSTR
        from eval.datasets.severity import INSTRUCTIONS as SEV_INSTR

        instr_by_family = {
            "banking77": B77_INSTR, "go_emotions": GOE_INSTR, "injection": INJ_INSTR,
            "severity": SEV_INSTR, "pubhealth": PUB_INSTR,
            # boolq/mmlu_pro instructions are per-row; those families use dict
            # states whose JSON is embedded verbatim - matched via the JSON blob
        }
        state_set = set()
        for p in HOLDOUT.glob("*.parquet"):
            for d in pq.read_table(p, columns=["state"]).to_pylist():
                s = str(d["state"])
                # parquet stores json.dumps(state); the prompt embeds raw str for
                # string states and the identical JSON string for dict states
                state_set.add(s if s.lstrip()[:1] in "{[" else json.loads(s))

        def extract_state(row) -> str | None:
            p = row["prompt"]
            if not p.startswith("State:\n"):
                return None
            body = p[len("State:\n") :]
            instr = instr_by_family.get(row["family"])
            if instr is None:  # per-row instruction: dict-state families
                i = body.find("\n\nQuestion: ")
                return body[:i] if i > 0 else None
            marker = f"\n\nQuestion: {instr}"
            i = body.rfind(marker)
            return body[:i] if i > 0 else None

        kept, dropped = [], 0
        for r in rows:
            st = extract_state(r)
            if st is not None and st in state_set:
                dropped += 1
            else:
                kept.append(r)
        rows = kept
        stats["n_decontam_dropped"] = dropped
    # 2% holdout split for train-time val loss only (never the frozen suite)
    val = rows[::50][: max(1, len(rows) // 50)]
    val_ids = {id(r) for r in val}
    train = [r for r in rows if id(r) not in val_ids]
    stats.update(n_train=len(train), n_val=len(val))
    return (train, val), stats  # type: ignore[return-value]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--mixture", default=str(MIXTURE))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None, help="cap optimizer steps (smoke)")
    ap.add_argument("--out", default="checkpoints/s1-v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-decontam", action="store_true")
    ap.add_argument("--force", action="store_true", help="start even with <8GB GPU free")
    a = ap.parse_args()

    if a.device == "cuda":
        free = gpu_free_gb()
        if free < 8 and not a.force:
            raise SystemExit(
                f"only {free:.1f} GB free on GPU:0 — llama-server is likely still up. "
                f"Kill it (or pass --force)."
            )
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    (train, val), stats = load_rows(Path(a.mixture), not a.no_decontam)
    print(json.dumps({"stage": "data", **stats}), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds_tr = DecisionSFT(train, tok, a.max_len)
    ds_val = DecisionSFT(val, tok, a.max_len)

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map=a.device if a.device == "cuda" else None
    )
    if a.device == "cpu":
        model = model.to("cpu")
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    per_rank = max(1, a.bs)
    steps_per_epoch = math.ceil(len(ds_tr) / (per_rank * a.accum))
    total_steps = steps_per_epoch * a.epochs if not a.steps else a.steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))
    logf = Path(a.out + "_log.jsonl")
    logf.parent.mkdir(parents=True, exist_ok=True)

    global_step, t0, run_loss = 0, time.time(), 0.0
    model.train()
    done = False
    for ep in range(a.epochs):
        perm = torch.randperm(len(ds_tr)).tolist()
        for j0 in range(0, len(perm), per_rank):
            idx = perm[j0 : j0 + per_rank]
            batch = {k: v.to(model.device) for k, v in collate([ds_tr[i] for i in idx], tok.pad_token_id).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.device == "cuda")):
                out = model(**batch)
            loss = out.loss / a.accum
            loss.backward()
            run_loss += float(out.loss)
            if (j0 // per_rank + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                rec = {"step": global_step, "epoch": ep, "loss": round(run_loss / a.accum, 4),
                       "lr": sched.get_last_lr()[0], "s": round(time.time() - t0, 1)}
                run_loss, t0 = 0.0, time.time()
                print(json.dumps(rec), flush=True)
                with logf.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                if a.steps and global_step >= a.steps:
                    done = True
                    break
        if done:
            break

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    print(json.dumps({"stage": "saved", "path": str(outdir), "steps": global_step}), flush=True)


if __name__ == "__main__":
    main()
