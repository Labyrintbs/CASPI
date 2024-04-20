
while [[ "$#" -gt 0 ]]; do
    case $1 in
    	--cuda) cuda="$2"; shift ;;
        --K) K="$2"; shift ;;
        --fold) fold=$2; shift ;;
        --metric) metric=$2; shift ;;
        --seed) seed=$2; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

ratio=5
#data_file=data_for_damd_reward_${K}.json
data_file=cntfact_data_for_damd_K_5_debug.json
#data_file=cntfact_data_for_damd_ratio_0.6.json
if [ $metric == 'soft' ]; then
  soft_acc=True
else
  soft_acc=False
fi

gen_per_epoch_report=True
enable_aspn=True
bspn_mode=bspn
enable_dst=False
use_true_curr_bspn=True
enable_cntfact=False
enable_debug=False
enable_contrast=False
enable_tensorboard=True
enable_multi_cntfact=False
enable_rl=False
enable_cntfact_reward=False
enable_contrast_reward=False
cntfact_penalty=-0.5
use_bcq=True
#train_bcq=True
#exp_path=experiments/20240314_all_debug_bcq_cntfact_bspn_reward_K_5_sd20240311_lr0.005_bs128_sp2_dc3_actTrue_0.05_hashbNElGUjJpMbDA
#bcq_model_path=experiments/20240314_all_debug_bcq_cntfact_bspn_reward_K_5_sd20240311_lr0.005_bs128_sp2_dc3_actTrue_0.05_hashbNElGUjJpMbDA/bcq_model.pkl
#bcq_model_path=experiments/20240319_all_debug_generate_bcq_train_K_5_sd20240316_lr0.005_bs128_sp3_dc3_actTrue_0.05_hashbgBEdUlfPVqA/bcq_model.pkl
# sample ratio 0.1
bcq_model_path=experiments/20240320_all_debug_bcq_train_K_5_sd20240316_lr0.005_bs128_sp3_dc3_actTrue_0.05_hashbgBEdUlfPVqA/bcq_model.pkl
early_stop_count=5


root_path=./damd_multiwoz

per_epoch_report_path=${root_path}/data/multi-woz-oppe/reward/20240322_bcq_report_${K}_${metric}_${fold}_ratio_${ratio}_dp.csv
dev_list=${root_path}/data/multi-woz-processed/rewardListFile_${K}_${fold}.json

exp_name=debug_bspn_cntfact_reward_K_${K}_fold_${fold}_metric_${metric}_seed_${seed}_CntfactRatio_${ratio}

log_file=${exp_name}.log
log_path=${root_path}/logs/${log_file}
echo 'To view log tail:'${log_path}

python  ${root_path}/model.py -mode train -cfg seed=$seed cuda_device=$cuda \
	exp_no=supervised_bcq_input_train_K_5_sampleRaito_0.1 batch_size=64 multi_acts_training=False \
	use_true_curr_bspn=${use_true_curr_bspn} \
	enable_aspn=${enable_aspn} \
	bspn_mode=${bspn_mode} \
	enable_dst=${enable_dst} \
	use_true_curr_bspn=${use_true_curr_bspn} \
	data_file=${data_file} \
	gen_per_epoch_report=${gen_per_epoch_report} \
	per_epoch_report_path=${per_epoch_report_path} \
	dev_list=${dev_list} \
	soft_acc=${soft_acc} \
	enable_cntfact=${enable_cntfact} \
	enable_debug=${enable_debug} \
	enable_contrast=${enable_contrast} \
	enable_tensorboard=${enable_tensorboard}\
	enable_multi_cntfact=${enable_multi_cntfact}\
	enable_rl=${enable_rl} \
	enable_cntfact_reward=${enable_cntfact_reward} \
	enable_contrast_reward=${enable_contrast_reward} \
	cntfact_penalty=${cntfact_penalty} \
	early_stop_count=${early_stop_count} \
	bcq_model_path=${bcq_model_path} \
	use_bcq=${use_bcq} \
	#exp_path=${exp_path} \
	#train_bcq=${train_bcq} \

