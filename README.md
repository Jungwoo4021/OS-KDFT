
<h4 align="right">
    <p>
        | <b>English</b> |
        <a href="https://github.com/Jungwoo4021/OS-KDFT/readme/README_ko.md">한국어</a> |
    </p>
</h4>

<div style="text-align: center; font-size: 2.5em;"> <b>OS-KDFT</div>
<h2 align="center">
    <p><b>Knowledge distillation</b> and <b>target task joint training</b> <br>for audio PLM compression and fine-tuning</p>
</h2>

<h3 align="left">
	<a href="https://www.isca-speech.org/archive/interspeech_2023/heo23_interspeech.html"><img src="https://img.shields.io/badge/DOI-10.21437/Interspeech.2023--605-blue" alt="Doi"></a>
	<br>
	<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=Python&logoColor=white"></a>
	<a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-08.html#rel-23-08"><img src="https://img.shields.io/badge/23.08-2496ED?style=for-the-badge&logo=Docker&logoColor=white"></a>
	<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"></a>
	<a href="https://huggingface.co/"><img src="https://github.com/Jungwoo4021/OS-KDFT/readme/icon_hugging_face.png"></a>
</h3>

# Introduction
Pytorch code for following paper:

* **Title** : One-Step Knowledge Distillation and Fine-Tuning in Using Large Pre-Trained Self-Supervised Learning Models for Speaker Verification (Accepted at Interspeech2023)
* **Autor** :  Jungwoo Heo, Chan-yeong Lim, Ju-ho Kim, Hyun-seo Shin, Ha-Jin Yu

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
@inproceedings{heo23_interspeech,
  author={Jungwoo Heo and Chan-yeong Lim and Ju-ho Kim and Hyun-seo Shin and Ha-Jin Yu},
  title={{One-Step Knowledge Distillation and Fine-Tuning in Using Large Pre-Trained Self-Supervised Learning Models for Speaker Verification}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={5271--5275},
  doi={10.21437/Interspeech.2023-605}
}
```
