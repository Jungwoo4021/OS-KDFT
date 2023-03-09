# One-Step Knowledge Distillation and Fine-Tuning in Using Large Pre-Trained Self-Supervised Learning Models for Speaker Verification

Pytorch code for following paper:

* **Title** : One-Step Knowledge Distillation and Fine-Tuning in Using Large Pre-Trained Self-Supervised Learning Models for Speaker Verification (Submitted at Interspeech2023)
* **Autor** :  Anonymous submission

# Abstract
The application of speech self-supervised learning (SSL) models has achieved remarkable performance in speaker verification (SV). However, there is a computational cost hurdle in employing them, which makes development and deployment difficult. Several studies have simply compressed SSL models through knowledge distillation (KD) without considering the target task. Consequently, these methods could not extract SV-tailored features. This paper suggests One-Step Knowledge Distillation and Fine-Tuning (OS-KDFT), which incorporates KD and fine-tuning (FT). We optimize a student model for SV during KD training to avert the distillation of inappropriate information for the SV. OS-KDFT could downsize Wav2Vec 2.0 based ECAPA-TDNN size by approximately 76.2%, and reduce the SSL model's inference time by 79% while presenting an EER of 0.98%. The proposed OS-KDFT is validated across VoxCeleb1 and VoxCeleb2 datasets and W2V2 and HuBERT SSL models. Experiments are available on our GitHub. 

# Prerequisites

## Environment Setting
* We used 'nvcr.io/nvidia/pytorch:22.01-py3' image of Nvidia GPU Cloud for conducting our experiments. 

* Python 3.8.12

* Pytorch 1.11.0+cu115

* Torchaudio 0.11.0+cu115

  

# Datasets

We used VoxCeleb1 and VoxCeleb2 datasets to evaluate our proposed method. Also, we employed MUSAN and RIR reverberation datasets for data augmentation. 

# Run experiment

Run main.py in scripts.

```python
> python main.py
```

### Set system arguments

First, you need to set system arguments. You can set arguments in `arguments.py`. Here is list of system arguments to set.

```python
1. 'path_log'	  : {YOUR_PATH}
	'path_log' is path of saving experiments.
	input type is str

	CAUTION! Don't set your path_log inside the experiment code path.
	'~/#00.Experiment_name/log_path' (X)
	'~/result/log_path'(O)

2. 'path_train'  : {YOUR_PATH}
	'path_train' is path where VoxCeleb2 train partition is stored.
	input type is str

3. 'path_test'  : {YOUR_PATH}
	'path_test' is path where VoxCeleb2 test partition is stored.
	input type is str

4. 'path_trials'  : {YOUR_PATH}
	'path_trials' is path where Vox1-O, Vox1-E, Vox1-H trials is stored.
	input type is str

5. 'path_musan'  : {YOUR_PATH}
	'path_musan' is path where MUSAN dataset is stored.
	input type is str

6. 'path_rir'  : {YOUR_PATH}
	'path_rir' is path where RIR reverberation dataset is stored.
	input type is str
```

### Additional logger

We have a basic logger that stores information in local. However, if you would like to use an additional online logger (wandb or neptune):

1. In `arguments.py`

```python
# Wandb: Add 'wandb_user' and 'wandb_token'
# Neptune: Add 'neptune_user' and 'neptune_token'
# input this arguments in "system_args" dictionary:
# for example
'wandb_user'   : 'user-name',
'wandb_token'  : 'WANDB_TOKEN',

'neptune_user'  : 'user-name',
'neptune_token' : 'NEPTUNE_TOKEN'
```

2. In `main.py`

```python
# Just remove "#" in logger

logger = LogModuleController.Builder(args['name'], args['project'],
        ).tags(args['tags']
        ).description(args['description']
        ).save_source_files(args['path_scripts']
        ).use_local(args['path_log']
        #).use_wandb(args['wandb_user'], args['wandb_token'] <- here
        #).use_neptune(args['neptune_user'], args['neptune_token'] <- here
        ).build()
```

# Citation

Please cite this paper if you make use of the code. 

```
@article{
}
```