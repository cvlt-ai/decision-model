#!/usr/bin/env bash
# s1-v6 eval: the TSI breadth line is the reason v6 exists, so the knowledge
# (mmlu_pro) + short-contrastive (VitaminC flip) axes lead, with regression checks after.
#  1. VitaminC flip probe (short-contrastive axis; TSI adds NLI/contrastive breadth)
#  2. Frozen 9-family suite @1024 (mmlu_pro is THE knowledge axis: v5 0.350, Jev 0.83)
#  3. JevBench public-231 @4096 + @1024 (does breadth help the shared yardstick?)
#  4. Offline calibration + consolidated comparison + commit.
set -uo pipefail
cd /home/user/decision-model

echo "===== [1/4] VitaminC flip probe (evidence-sensitivity axis, headline) ====="
.venv/bin/python scripts/vitaminc_flip_probe.py --ckpt checkpoints/s1-v6 \
  --base Qwen/Qwen3.5-4B --device cuda:0 --families 200 \
  --out results/vitaminc_flip_s1v6.json 2>&1 | tail -20

echo "===== [2/4] Frozen 9-family suite @1024 (regression check) ====="
.venv/bin/python -m eval.run_eval --model readout --ckpt checkpoints/s1-v6 \
  --base Qwen/Qwen3.5-4B --device cuda:0 --key-batch 8 \
  --families boolq,banking77,go_emotions,mmlu_pro,pubhealth,injection,severity,baserate,negation \
  --limit 300 --raw results/raw_s1v6_frozen.jsonl --fresh \
  --out results/s1v6_frozen_300.json 2>&1 | grep -E "acc=|macro|rror|Traceback" | tail -15

echo "===== [3/4] JevBench public-231 @4096 (headline) ====="
.venv/bin/python scripts/jevbench_run.py --ckpt checkpoints/s1-v6 --device cuda:0 \
  --key-batch 8 --max-len 4096 --out results/jevbench_s1v6_public231_4096.json 2>&1 \
  | grep -E "\[|OVERALL|ECE|wrote" | tail -8
echo "===== [3/4] JevBench public-231 @1024 (delta) ====="
.venv/bin/python scripts/jevbench_run.py --ckpt checkpoints/s1-v6 --device cuda:0 \
  --key-batch 8 --max-len 1024 --out results/jevbench_s1v6_public231.json 2>&1 \
  | grep -E "\[|OVERALL|ECE|wrote" | tail -8

echo "===== [4/4] Calibration + comparison ====="
.venv/bin/python scripts/calibrate.py --raw results/raw_s1v6_frozen.jsonl --tag s1v6 2>&1 | tail -3
.venv/bin/python scripts/compare_models.py > results/compare_models_latest.txt 2>&1

echo "===== v6 summary ====="
.venv/bin/python - <<'PY'
import json, os
def L(p): return json.JSONDecoder().raw_decode(open(p).read())[0] if os.path.exists(p) else None
def J(p): return json.load(open(p)) if os.path.exists(p) else None
v6f=L("results/vitaminc_flip_s1v6.json"); v3f=L("results/vitaminc_flip_s1v3.json")
if v6f and v3f:
    print(f"VITAMINC FLIP  v3: NEI={v3f['per_class']['NOT ENOUGH INFO']['acc']:.3f} lazy={v3f['lazy_rate']}  "
          f"|  v6: NEI={v6f['per_class']['NOT ENOUGH INFO']['acc']:.3f} lazy={v6f['lazy_rate']}  "
          f"acc {v3f['accuracy']}->{v6f['accuracy']}")
# THE knowledge axis: mmlu_pro v5 -> v6, plus macro regression
v5f=J("results/s1v5_frozen_300.json"); v6fz=J("results/s1v6_frozen_300.json")
if v5f and v6fz:
    for fam in ["mmlu_pro","pubhealth"]:
        a=v5f["families"].get(fam,{}).get("accuracy"); b=v6fz["families"].get(fam,{}).get("accuracy")
        print(f"FROZEN {fam:11} v5={a} -> v6={b}")
    print(f"FROZEN macro      v5={v5f['macro']['accuracy']:.3f} -> v6={v6fz['macro']['accuracy']:.3f}  "
          f"negation {v5f['macro']['negation_mean_violation']} -> {v6fz['macro']['negation_mean_violation']}")
for tag,p in [("v6@1024","results/jevbench_s1v6_public231.json"),
              ("v6@4096","results/jevbench_s1v6_public231_4096.json")]:
    d=L(p)
    if d:
        o=d["overall"]
        print(f"{tag} JevBench {o['correct']}/{o['n']} = {o['accuracy']:.4f} "
              f"(hard {d['tiers']['hard']['correct']}/111)")
print("REF: v5 JevBench 0.719/0.749  AlexWortega 0.814  Jev 0.866")
PY
echo "===== DONE v6 eval ====="
