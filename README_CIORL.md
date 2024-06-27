**CIORL**

This is the instruction for CIORL implementation

## Dependency 

pip install -r ./requirements.txt

SentenceTransformer version: all-MiniLM-L6-v2 (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 

- path: /home/fanghongming/all-MiniLM-L6-v2 in beijing server

## Data setup
Please follow CASPI and DAMD's Readme first to split dataset and preprocess them.

IMPORTANT:
config setting (based on CASPI):

- path: /damd_multiwoz/config.py        

modified CASPI for CIORL preprocess: 
- path: /damd_multiwoz/preprocess.py

related configs:
- self.preprocess_model_path (Sentence Transformers path)
- self.topk_cntfact = 5 (Top K cntfact number)

Run command for cntfact data generation
```console
python preprocess.py
```

## Experiments

following CASPI: 

**Create K-fold datasets**

Please choose appropriate number of folds. In our work, we use 10 folds. larger the number of folds, larger number of model needs to be trained. For quick turn around smaller number of folds with marginal loss in performance.
```console
python CreateKFoldDataset.py --seed 111 --folds 10
```

**Supervised Learning**
This will run 10 seeds the same way as CASPI.
```console
bash run_CASPI_baseline.sh
```

**Contrast Learning**
This will run 10 seeds using contrast learning only.
```console
bash run_contrast.sh
```

**Reinforcement Learning**
First run 
```console
bash ./damd_multiwoz/scripts/gen_reward_rollout_gen_bcq_buffer.sh --cuda 0 --K 10 --fold 0 --metric soft --seed 20240316
```
to generate rl transition buffer

Then use 
```console
bash ./damd_multiwoz/scripts/gen_reward_rollout_gen_bcq_buffer.sh --cuda 0 --K 10 --fold 0 --metric soft --seed 20240316
```
to train the bcq model

Finally use
```console
bash run_bcq.sh
```
to train CIORL using the bcq in the prev step. 