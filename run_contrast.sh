#!/bin/bash
base_seed=68690

for fold in {9..9}; do
    seed=$((base_seed + fold))
	bash ./damd_multiwoz/scripts/gen_reward_rollout_contrast.sh --cuda 0 --K 10 --fold "$fold" --metric soft --seed "$seed"
done

