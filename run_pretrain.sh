#!/bin/bash
base_seed=68690

for fold in {0..9}; do
    seed=$((base_seed + fold))
	bash ./damd_multiwoz/scripts/gen_reward_rollout.sh --cuda 1 --K 10 --fold "$fold" --metric soft --seed "$seed"
done

