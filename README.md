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

> If you want to make new 10-fold information, just run `/GTZAN/k_fold.py` in GTZAN dataset or `Melon/k_fold.py` in Melon Playlist dataset.

# Run experiment

Go into experiment folder what you want to run.

### Set system arguments

First, you need to set system arguments. You can set arguments in `#00.Experiements/arguments.py`. Here is list of system arguments to set.

```python
1. 'path_log'	  : {YOUR_PATH}
	'path_log' is path of saving experiments.
	input type is str

	CAUTION! Don't set your path_log inside the experiment code path.
	'~/#00.Experiment_name/log_path' (X)
	'~/result/log_path'(O)

2. 'path_gtzan'  : {YOUR_PATH}
	'path_gtzan' is path where GTZAN dataset is saved.
	input type is str

	# ex) '/data/GTZAN'

3. 'kfold_ver'   : {K-FOLD_VER}
	'kfold_ver' is k number of k-Fold.
	You can use [1,2,3,4,5,6,7,8,9,10] numbers.
	input type is int

	# ex) 1
```

### Additional logger

We have a basic logger that stores information in local. However, if you would like to use an additional online logger (wandb or neptune):

1. In `arguments.py`

```python
# Wandb: Add 'wandb_user' and 'wandb_token'
# Neptune: Add 'neptune_user' and 'neptune_token'
# input this arguments in "system_args" dictionary:
# for example
'wandb_user'   : 'Hyun-seo',
'wandb_token'  : 'WANDB_TOKEN',
```

2. In `main.py`

```python
# Just remove "#" in logger

logger = LogModuleController.Builder(args['name'], args['project'],
        ).tags(args['tags']
        ).description(args['description']
        ).save_source_files(args['path_scripts']
        ).use_local(args['path_log']
        ).use_wandb(args['wandb_user'], args['wandb_token']
        ).build()
```

### Run experiment code

```python
# And just run main.py
python main.py
```

We adopt 10-fold cross-validation to get system performance (accuracy). So you need to change `kfold_ver` and re-run experiment code.



# Citation

Please cite this paper if you make use of the code. 

```
@article{
}
```

# Reference
We implemented the BBNN system with reference to [here]( https://arxiv.org/pdf/1901.08928.pdf ). The original BBNN code can be found [here]( https://github.com/CaifengLiu/music-genre-classification ).
