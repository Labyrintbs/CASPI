#!/bin/bash
base_seed=68690

for fold in {0..9}; do
    seed=$((base_seed + fold))
	bash ./damd_multiwoz/scripts/caspi_damd.sh --cuda 0 --seed "$seed" --K 10 --gamma 0.0 --policy_loss L_det --action_space act --metric soft --train_e2e False 
done

