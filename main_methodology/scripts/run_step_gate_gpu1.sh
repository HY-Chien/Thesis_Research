#!/bin/bash
# Step-gate signal-swap overnight chain, GPU1: probe hotpotqa+popqa ->
# wait for manifest -> synthesis shard 1.
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis
cd /home/S113065528/Research/main_methodology
set -e

for ds in hotpotqa popqa; do
  CUDA_VISIBLE_DEVICES=1 python scripts/probe_step_gate_battery.py \
    --trace results/${ds}_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl \
    --out results/step_gate_battery_${ds}.jsonl --resume \
    >> scripts/step_gate_probe_${ds}.log 2>&1
done
touch scripts/step_gate_gpu1_probe.done

while [ ! -f scripts/step_gate_manifest.done ]; do sleep 30; done

CUDA_VISIBLE_DEVICES=1 python scripts/gen_manifest_synthesis.py \
  --shard 1 --num-shards 2 --out results/step_gate_synth_shard1.jsonl --resume \
  >> scripts/step_gate_synth_shard1.log 2>&1
touch scripts/step_gate_gpu1_synth.done
echo "GPU1 CHAIN DONE"
