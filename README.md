
<h4 align="right">
    <p>
        | <b>English</b> |
        <a href="https://github.com/Jungwoo4021/OS-KDFT/blob/main/readme/README_ko.md">한국어</a> |
    </p>
</h4>

<h1 align="center">
    <b>OS-KDFT</b>
</h1>

<h2 align="center">
    <b>Knowledge distillation</b> and <b>target task joint training</b> <br>for audio PLM compression and fine-tuning
</h2>

<h3 align="left">
	<p>
	<a href="https://www.isca-speech.org/archive/interspeech_2023/heo23_interspeech.html"><img src="https://img.shields.io/badge/DOI-10.21437/Interspeech.2023--605-blue" alt="Doi"></a>
	<br>
	<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=Python&logoColor=white"></a>
	<a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-08.html#rel-23-08"><img src="https://img.shields.io/badge/23.08-2496ED?style=for-the-badge&logo=Docker&logoColor=white"></a>
	<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"></a>
	<a href="https://huggingface.co/"><img src="https://github.com/Jungwoo4021/OS-KDFT/blob/main/readme/icon_hugging_face.png?raw=true"></a>
	</p>
</h3>

# Update log
(In progress) 
* (~2024.02) Develop an improved version of OS-KDFT
* (~2024.02) Check on other tasks (speech recognition, keyword spotting, etc.)

(Done)
* (2024.01.15) Upload evaluation scripts


# About OS-KDFT
### Summary
This repository offers source code for following paper:

* **Title** : One-Step Knowledge Distillation and Fine-Tuning in Using Large Pre-Trained Self-Supervised Learning Models for Speaker Verification (Accepted at Interspeech2023)
* **Autor** :  Jungwoo Heo, Chan-yeong Lim, Ju-ho Kim, Hyun-seo Shin, Ha-Jin Yu

We provide experiment scripts, trained model, and training logs. 

### Paper abstract
The application of speech self-supervised learning (SSL) models has achieved remarkable performance in speaker verification (SV). However, there is a computational cost hurdle in employing them, which makes development and deployment difficult. Several studies have simply compressed SSL models through knowledge distillation (KD) without considering the target task. Consequently, these methods could not extract SV-tailored features. This paper suggests One-Step Knowledge Distillation and Fine-Tuning (OS-KDFT), which incorporates KD and fine-tuning (FT). We optimize a student model for SV during KD training to avert the distillation of inappropriate information for the SV. OS-KDFT could downsize Wav2Vec 2.0 based ECAPA-TDNN size by approximately 76.2%, and reduce the SSL model's inference time by 79% while presenting an EER of 0.98%. The proposed OS-KDFT is validated across VoxCeleb1 and VoxCeleb2 datasets and W2V2 and HuBERT SSL models. 

# What can I do in this repository?
You can get the experimental code via hyperlinks. 
<br> Note that we provide our **trained model weights** and **training logs** (such as loss, validation results) for re-implementation. You can find these in 'params' folder stored in each 'only evaluation' folder.  

1. HuBERT compression in speaker verification, EER 4.75% in VoxCeleb1 (<a href="https://github.com/Jungwoo4021/OS-KDFT/tree/main/scripts/HuBERT_DistilHuBERT_SV_Vox1/train">train & evaluation</a>, <a href="https://github.com/Jungwoo4021/OS-KDFT/tree/main/scripts/HuBERT_DistilHuBERT_SV_Vox1/only_eval">only evaluation</a>)
2. WavLM compression in speaker verification, EER 4.25% in VoxCeleb1 (<a href="https://github.com/Jungwoo4021/OS-KDFT/tree/main/scripts/WavLM_DistHuBERT_SV_Vox1/train">train & evaluation</a>, <a href="https://github.com/Jungwoo4021/OS-KDFT/tree/main/scripts/WavLM_DistHuBERT_SV_Vox1/only_eval">only evaluation</a>)

# How to use?
## 1. Set environment

### 1-1. Open NVIDIA-docker

<a href="https://github.com/Jungwoo4021/OS-KDFT/Dockerfile"><img src="https://img.shields.io/badge/DOCKER FILE-2496ED?style=for-the-badge&logo=Docker&logoColor=white"></a>
```
Docker file summary

Docker
    nvcr.io/nvidia/pytorch:23.08-py3 

Python
    3.8.12

Pytorch 
    2.1.0a0+29c30b1

Torchaudio 
    2.0.1
```
(We conducted experiment using 2 or 4 NVIDIA RTX A5000 GPUs)

### 1-2. Prepare datasets

Depending on the task you want to perform, you'll need the following datasets.

* Speaker verification: (VoxCeleb1) or (VoxCeleb2, MUSAN, RIR reverberation)
* Keyword spotting: To be updated


## 2. Run expriment
### 2-1. Download script

You can get the experimental code via the hyperlinks in the "What can I do in this repository?" section.

### 2-2. Set system arguments

Set experimental arguments in `arguments.py` file. Here is list of system arguments to set.

```python
1. 'usable_gpu': {YOUR_PATH} # ex) '0,1,2,3'
	'path_log' is path of saving experiments.
	input type is str

2. 'path_...': {YOUR_PATH}
	'path_...' is path where ... dataset is stored.
	input type is str
```

&nbsp;

### About logger

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
# Just remove "#" in logger which you use

logger = LogModuleController.Builder(args['name'], args['project'],
        ).tags(args['tags']
        ).description(args['description']
        ).save_source_files(args['path_scripts']
        ).use_local(args['path_log']
        #).use_wandb(args['wandb_user'], args['wandb_token'] <- here
        #).use_neptune(args['neptune_user'], args['neptune_token'] <- here
        ).build()
```
### 2-3. Run!

Just run main.py in scripts!

```python
> python main.py
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
