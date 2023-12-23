
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

ratio=0.4
#data_file=data_for_damd_reward_${K}.json
data_file=cntfact_data_for_damd_ratio_0.4.json

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
enable_cntfact=True

root_path=./damd_multiwoz

per_epoch_report_path=${root_path}/data/multi-woz-oppe/reward/bspn_cntfact_reward_report_${K}_${metric}_${fold}_ratio_${ratio}_dp.csv
dev_list=${root_path}/data/multi-woz-processed/rewardListFile_${K}_${fold}.json

exp_name=bspn_cntfact_reward_K_${K}_fold_${fold}_metric_${metric}_seed_${seed}_CntfactRatio_${ratio}

log_file=${exp_name}.log
log_path=${root_path}/logs/${log_file}
echo 'To view log tail:'${log_path}

python  ${root_path}/model.py -mode train -cfg seed=$seed cuda_device=$cuda \
	exp_no=no_aug_with_bs_with_cntfact_bspn batch_size=128 multi_acts_training=False \
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
	enable_cntfact=${enable_cntfact}

