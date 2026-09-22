#!/usr/bin/env bash
# Refuse to touch a GPU that the live inference server is holding.
# Usage: scripts/gpu_free.sh <gpu_index> <needed_mib>   -> exit 0 if safe
set -euo pipefail
i=${1:?usage: gpu_free.sh <gpu_index> <needed_mib>}
need=${2:?usage: gpu_free.sh <gpu_index> <needed_mib>}

total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$i")
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i")
free=$(( total - used ))

# Anything holding >1GB right now is a live workload, not our own test.
holders=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
          | awk -F', ' -v g="$i" '$2 > 1024 {print $1"("$2"Mib)"}' | paste -sd' ' -)

echo "gpu${i}: total=${total}MiB used=${used}MiB free=${free}MiB need=${need}MiB"
if [ -n "${holders}" ]; then
  echo "gpu${i}: live processes holding >1GiB: ${holders}"
fi
if [ "$free" -ge "$need" ]; then
  echo "gpu${i}: READY"
  exit 0
fi
echo "gpu${i}: DEFERRED — not enough free VRAM. Do NOT kill llama-server; ask the operator." >&2
exit 1
