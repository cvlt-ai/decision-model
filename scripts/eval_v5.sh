#!/usr/bin/env bash
# s1-v5 eval: the c2d contrastive line is the reason v5 exists, so the
# evidence-sensitivity axis (VitaminC flip probe) leads, with regression checks after.
#  1. VitaminC flip probe @ v3-protocol (headline: vs v3 baseline NEI 0.308 / lazy 0.175)
#  2. Frozen 9-family suite @1024 (regression check vs v3 macro 0.727, negation 0.030)
#  3. JevBench public-231 @4096 + @1024 (does contrast data help the shared yardstick?)
#  4. Offline calibration + consolidated comparison + commit.
set -uo pipefail
cd /home/user/decision-model

echo "===== [1/4] VitaminC flip probe (evidence-sensitivity axis, headline) ====="
.venv/bin/python scripts/vitaminc_flip_probe.py --ckpt checkpoints/s1-v5 \
  --base Qwen/Qwen3.5-4B --device cuda:0 --families 200 \
  --out results/vitaminc_flip_s1v5.json 2>&1 | tail -20

echo "===== [2/4] Frozen 9-family suite @1024 (regression check) ====="
.venv/bin/python -m eval.run_eval --model readout --ckpt checkpoints/s1-v5 \
  --base Qwen/Qwen3.5-4B --device cuda:0 --key-batch 8 \
  --families boolq,banking77,go_emotions,mmlu_pro,pubhealth,injection,severity,baserate,negation \
  --limit 300 --raw results/raw_s1v5_frozen.jsonl --fresh \
  --out results/s1v5_frozen_300.json 2>&1 | grep -E "acc=|macro|rror|Traceback" | tail -15

echo "===== [3/4] JevBench public-231 @4096 (headline) ====="
.venv/bin/python scripts/jevbench_run.py --ckpt checkpoints/s1-v5 --device cuda:0 \
  --key-batch 8 --max-len 4096 --out results/jevbench_s1v5_public231_4096.json 2>&1 \
  | grep -E "\[|OVERALL|ECE|wrote" | tail -8
echo "===== [3/4] JevBench public-231 @1024 (delta) ====="
.venv/bin/python scripts/jevbench_run.py --ckpt checkpoints/s1-v5 --device cuda:0 \
  --key-batch 8 --max-len 1024 --out results/jevbench_s1v5_public231.json 2>&1 \
  | grep -E "\[|OVERALL|ECE|wrote" | tail -8

echo "===== [4/4] Calibration + comparison ====="
.venv/bin/python scripts/calibrate.py --raw results/raw_s1v5_frozen.jsonl --tag s1v5 2>&1 | tail -3
.venv/bin/python scripts/compare_models.py > results/compare_models_latest.txt 2>&1

echo "===== v5 summary ====="
.venv/bin/python - <<'PY'
import json, os
def L(p): return json.JSONDecoder().raw_decode(open(p).read())[0] if os.path.exists(p) else None
v5=L("results/vitaminc_flip_s1v5.json"); v3=L("results/vitaminc_flip_s1v3.json")
if v5 and v3:
    print(f"VITAMINC FLIP  v3: NEI={v3['per_class']['NOT ENOUGH INFO']['acc']:.3f} lazy={v3['lazy_rate']}  "
          f"|  v5: NEI={v5['per_class']['NOT ENOUGH INFO']['acc']:.3f} lazy={v5['lazy_rate']}  "
          f"acc {v3['accuracy']}->{v5['accuracy']}")
for tag,p in [("v5@1024","results/jevbench_s1v5_public231.json"),
              ("v5@4096","results/jevbench_s1v5_public231_4096.json")]:
    d=L(p)
    if d:
        o=d["overall"]
        print(f"{tag} JevBench {o['correct']}/{o['n']} = {o['accuracy']:.4f} "
              f"(hard {d['tiers']['hard']['correct']}/111)")
print("REF: v3 JevBench 0.675/0.693  AlexWortega 0.814  Jev 0.866")
PY
echo "===== DONE v5 eval ====="
