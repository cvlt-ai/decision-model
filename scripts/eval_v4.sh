#!/usr/bin/env bash
# s1-v4 eval: the JevBench hard tier is the reason v4 exists, so it leads.
#  1. JevBench public-231 at 4096 (headline: vs v3 0.693, AlexWortega 0.814, Jev 0.866)
#  2. JevBench public-231 at 1024 (for the 1024-vs-4096 delta, same as v3 protocol)
#  3. Frozen 9-family suite at 1024 (regression check vs v3 macro 0.727)
#  4. Offline calibration + consolidated comparison + commit.
set -uo pipefail
cd /home/user/decision-model

echo "===== [1/4] JevBench public-231 @4096 (headline) ====="
.venv/bin/python scripts/jevbench_run.py --ckpt checkpoints/s1-v4 --device cuda:0 \
  --key-batch 8 --max-len 4096 --out results/jevbench_s1v4_public231_4096.json 2>&1 \
  | grep -E "\[|OVERALL|ECE|wrote" | tail -8

echo "===== [2/4] JevBench public-231 @1024 (delta vs v3 protocol) ====="
.venv/bin/python scripts/jevbench_run.py --ckpt checkpoints/s1-v4 --device cuda:0 \
  --key-batch 8 --max-len 1024 --out results/jevbench_s1v4_public231.json 2>&1 \
  | grep -E "\[|OVERALL|ECE|wrote" | tail -8

echo "===== [3/4] Frozen 9-family suite @1024 (regression check) ====="
.venv/bin/python -m eval.run_eval --model readout --ckpt checkpoints/s1-v4 \
  --base Qwen/Qwen3.5-4B --device cuda:0 --key-batch 8 \
  --families boolq,banking77,go_emotions,mmlu_pro,pubhealth,injection,severity,baserate,negation \
  --limit 300 --raw results/raw_s1v4_frozen.jsonl --fresh \
  --out results/s1v4_frozen_300.json 2>&1 | grep -E "acc=|macro|rror|Traceback" | tail -15

echo "===== [4/4] Calibration + comparison ====="
.venv/bin/python scripts/calibrate.py --raw results/raw_s1v4_frozen.jsonl --tag s1v4 2>&1 | tail -3
.venv/bin/python scripts/compare_models.py > results/compare_models_latest.txt 2>&1
.venv/bin/python - <<'PY'
import json
def L(p): return json.JSONDecoder().raw_decode(open(p).read())[0]
a=L("results/jevbench_s1v4_public231.json"); b=L("results/jevbench_s1v4_public231_4096.json")
print("\n=== s1-v4 JevBench public-231 ===")
for tag,d in [("v4@1024",a),("v4@4096",b)]:
    o=d["overall"]
    print(f"  {tag:9} {o['correct']}/{o['n']} = {o['accuracy']:.4f}  "
          f"(std {d['tiers']['standard']['correct']}/72 easy {d['tiers']['easy']['correct']}/48 "
          f"hard {d['tiers']['hard']['correct']}/111)")
print("  REF: v3@1024=0.675 v3@4096=0.693  AlexWortega=0.814  Jev=0.866")
PY
echo "===== DONE v4 eval ====="
