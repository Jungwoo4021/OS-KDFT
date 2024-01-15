import os
import random
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .metric import calculate_EER
from ..util import all_gather, synchronize
from ..dataset import SV_Trial, SV_EnrollmentItem
from .score_calibration import QMF

def SV_enrollment(framework, loader, use_TTA=False, run_on_ddp=False):
    '''Extract embeddings for speaker verification evaluation. 

    Params
        framework(Framework): 
        loader(DataLoader): torch dataloader that gives (utt, key). 
        TTA: set True if you use TTA
        run_on_ddp: set True if you use Distribution Data Parallel
    
    Return
        embedding_dict(dict): 
    '''
    framework.eval()
    
    # enrollment
    keys = []
    embeddings = []
    embedding_dict = {}
    with tqdm(total=len(loader), ncols=90) as pbar, torch.set_grad_enabled(False):
        for x, key in loader:
            # to GPU
            if run_on_ddp:
                x = x.to(dtype=torch.float32, device=framework.device, non_blocking=True)
            else:
                x = x.to(dtype=torch.float32, device=framework.device)
            
            # feed forward
            if use_TTA:
                batch, num_seg, _ = x.size()
                x = x.view(-1, x.size(-1))
                x = framework(x).to('cpu')
                x = x.view(batch, num_seg, -1)            
            else:
                x = framework(x).to('cpu')
            
            # append
            for i in range(x.size(0)):
                keys.append(key[i])
                embeddings.append(x[i])
            
            pbar.update(1)
    
    if run_on_ddp:
        synchronize()

        # gather
        keys = all_gather(keys)
        embeddings = all_gather(embeddings)

    # re-cast: list -> dict
    for i in range(len(keys)):
        embedding_dict[keys[i]] = embeddings[i]
            
    return embedding_dict

def _calculate_cos_sim(trials, embeddings, TTA=False):
    buffer = [[], []]
    for item in trials:
        buffer[0].append(embeddings[item.key1])
        buffer[1].append(embeddings[item.key2])
    
    if TTA:
        batch = len(trials)
        num_seg = buffer[0][0].size(0)
        b1 = torch.stack(buffer[0], dim=0).view(batch, num_seg, -1)
        b2 = torch.stack(buffer[1], dim=0).view(batch, num_seg, -1)
        b1 = b1.repeat(1, num_seg, 1).view(batch * num_seg * num_seg, -1)
        b2 = b2.repeat(1, 1, num_seg).view(batch * num_seg * num_seg, -1)

        cos_sims = F.cosine_similarity(b1, b2)
        cos_sims = cos_sims.view(batch, num_seg * num_seg)
        cos_sims = cos_sims.mean(dim=1)
        
        return cos_sims
    else:
        b1 = torch.stack(buffer[0], dim=0)
        b2 = torch.stack(buffer[1], dim=0)
        cos_sims = F.cosine_similarity(b1, b2)
    
        return cos_sims
    
def test_SV_EER(trials, mono_embedding=None, multi_embedding=None, get_score=False):
    '''Calculate EER for test speaker verification performance.
    This support 3 options:
        1. Using only 1 embedding in enrollment and test
        2. Using multiple embeddings in enrollment and test
        3. Perform 1 and 2 and mean their cosine_similarity score
    
    Param
        trials(list): list of SV_Trial (it contains key1, key2, label) 
        mono_embedding(dict): embedding dict extracted from single utterance
        multi_embedding(dict): embedding dict extracted from multi utterance, such as TTA
        get_score(bool): if True, return cos_sim scores 
    
    Return
        eer(float)
    '''
    labels = []
    cos_sims_mono = []
    cos_sims_multi = []

    count = 0
    for i in range(len(trials) // 1000):
        # split trials for saving RAM 
        if i == (len(trials) // 1000 - 1) and (len(trials) % 1000) != 0:
            s = i * 1000
            sub_trials = trials[s:]
        else:
            s = i * 1000
            e = (i + 1) * 1000
            sub_trials = trials[s:e]
        
        # label
        for item in sub_trials:
            labels.append(item.label)
        
        # cosine similarity
        if mono_embedding is not None:
            cos_sims_mono += _calculate_cos_sim(sub_trials, mono_embedding)
        
        if multi_embedding is not None:
            cos_sims_multi += _calculate_cos_sim(sub_trials, multi_embedding, TTA=True)

    if multi_embedding is None:
        cos_sims = cos_sims_mono
    elif mono_embedding is None:
        cos_sims = cos_sims_multi        
    else:
        cos_sims = [(cos_sims_mono[i] + cos_sims_multi[i]) / 2 for i in range(len(cos_sims_mono))]
    eer = calculate_EER(cos_sims, labels)
    
    if get_score:
        return eer, cos_sims
    else:
        return eer

def adaptive_s_norm(path, dataset, framework, trials, eval_embeddings, num_spk=600, path_cohort_set=None, cross_select=False, top_n=300, get_score=False):
    # sample cohort set
    if path_cohort_set is None:
        path_cohort_set = f'{path}/cohort_set.txt'
        sample_cohort_set(path_cohort_set, dataset, num_spk)
    
    # enrollment
    cohort_embeddings = enrollment_cohort_embedding(framework, path_cohort_set)
    
    # cosine score
    score_enroll_test, score_enroll_cohort, score_test_cohort = calculate_cohort_score(path, trials, eval_embeddings, cohort_embeddings)
    
    # asnorm
    output_scores = asnorm_score(score_enroll_test, score_enroll_cohort, score_test_cohort, cross_select, top_n, output_statistics=True)
    
    # EER
    scores = []
    labels = []
    for i in range(len(trials)):
        scores.append(output_scores[i][-1])
        labels.append(trials[i].label)
    eer = calculate_EER(scores, labels)
    
    if get_score:
        return eer, output_scores
    else:
        return eer
    
def sample_cohort_set(path, dataset, num_spk=600):
    '''Make cohort.txt used for ASNorm.
    --------------------------
    Example of output file
    spk1 /home/data/wav1 /home/data/wav2 /home/data/wav3
    spk2 /home/data/wav4 /home/data/wav5
    spk3 /home/data/wav6 /home/data/wav7 /home/data/wav8 /home/data/wav9
    --------------------------
    '''
    # select speaker
    cohort_dict = {}
    for item in dataset:
        try:
            cohort_dict[item.speaker].append(item.path)
        except:
            cohort_dict[item.speaker] = [item.path]  
    selected_speaker = random.sample(list(cohort_dict.keys()), num_spk)
    
    # save txt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, 'w')
    for spk in selected_speaker:
        line = f'{spk}'
        for wav in cohort_dict[spk]:
            line += f' {wav}'
        line += '\n'
        f.write(line)
    f.close()

class WavLoader(Dataset):
    def __init__(self, items):
        self.items = items
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # sample
        item = self.items[index]
        
        # read wav
        wav, _ = sf.read(item.path)
        
        return wav, item.key
    
def enrollment_cohort_embedding(framework, path_cohort_set):
    '''Enrollment cohort embedding
    
    Return
        embeddings(dict)
    '''
    # read txt 
    items = []
    embeddings = {}
    f = open(path_cohort_set, 'r')
    for line in f.readlines():
        line = line.replace('\n', '')
        strI = line.split(' ')
        speaker = strI[0]
        embeddings[speaker] = []
        items += [SV_EnrollmentItem(key=speaker, path=file) for file in strI[1:]]
        
    # data loader
    loader = DataLoader(
        WavLoader(items),
        num_workers=10,
        batch_size=1,
        pin_memory=True,
    )
    
    # enrollment
    framework.eval()
    with torch.set_grad_enabled(False):
        for wav, key in tqdm(loader, ncols=90, desc='cohort enrollment'):
            wav = wav.to(dtype=torch.float32, device=framework.device)
            embedding = framework(wav)[0]
            embedding = embedding.to('cpu')
            embeddings[key[0]].append(embedding)
            torch.cuda.empty_cache()
    
    # mean
    for key in embeddings.keys():
        embeddings[key] = torch.stack(embeddings[key], dim=0).mean(dim=0)
    
    return embeddings

def calculate_cohort_score(path, trials, eval_embeddings, cohort_embeddings):
    '''Make 'score_enroll_test.txt', 'score_enroll_cohort.txt', 'score_test_cohort.txt' files. 
    
    Return

    -----------------
    Contest example
    key1 key2 0.845145
    key3 key4 0.471515
    -----------------
    '''
    # save path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    enroll_test = { 'enroll': [], 'test': [], 'score': [] }
    enroll_cohort = { 'enroll': [], 'cohort': [], 'score': [] }
    test_cohort = { 'test': [], 'cohort': [], 'score': [] }
    
    cohort_keys = list(cohort_embeddings.keys())
    cohort_embedding_buffer = torch.stack([cohort_embeddings[k] for k in cohort_keys], dim=0)
    for item in tqdm(trials, ncols=90, desc='save cohort score'):
        key_enroll = item.key1
        key_test = item.key2
        
        # cosine_score (enrollment <-> test)
        enroll_embed = eval_embeddings[key_enroll]
        test_embed = eval_embeddings[key_test]
        s = F.cosine_similarity(enroll_embed.view(1, -1), test_embed.view(1, -1)).item()
        enroll_test['enroll'].append(key_enroll)
        enroll_test['test'].append(key_test)
        enroll_test['score'].append(s)
        
        # cosine_score (enrollment <-> cohort)
        buffer = enroll_embed.repeat(len(cohort_keys), 1)
        scores = F.cosine_similarity(buffer, cohort_embedding_buffer).tolist()
        for i in range(len(cohort_keys)):
            enroll_cohort['enroll'].append(key_enroll)
            enroll_cohort['cohort'].append(cohort_keys[i])
            enroll_cohort['score'].append(scores[i])
            
        # cosine_score (test <-> cohort)
        buffer = test_embed.repeat(len(cohort_keys), 1)
        scores = F.cosine_similarity(buffer, cohort_embedding_buffer).tolist()
        for i in range(len(cohort_keys)):
            test_cohort['test'].append(key_test)
            test_cohort['cohort'].append(cohort_keys[i])
            test_cohort['score'].append(scores[i])
    
    return enroll_test, enroll_cohort, test_cohort
    
def asnorm_score(f_enroll_test, f_enroll_cohort, f_test_cohort, cross_select=False, top_n=300, output_statistics=False):
    input_score = pd.DataFrame(f_enroll_test)
    enroll_cohort_score = pd.DataFrame(f_enroll_cohort)
    test_cohort_score = pd.DataFrame(f_test_cohort)
    
    output_score = []

    enroll_cohort_score.sort_values(by="score", ascending=False, inplace=True)
    test_cohort_score.sort_values(by="score", ascending=False, inplace=True)

    if cross_select:
        enroll_top_n = enroll_cohort_score.groupby("enroll").head(top_n)[["enroll", "cohort"]]
        test_group = pd.merge(pd.merge(input_score[["enroll", "test"]], enroll_top_n, on="enroll"), 
                              test_cohort_score, on=["test", "cohort"]).groupby(["enroll", "test"])

        test_top_n = test_cohort_score.groupby("test").head(top_n)[["test", "cohort"]]
        enroll_group = pd.merge(pd.merge(input_score[["enroll", "test"]], test_top_n, on="test"), 
                                enroll_cohort_score, on=["enroll", "cohort"]).groupby(["enroll", "test"])
    else:
        enroll_group = enroll_cohort_score.groupby("enroll").head(top_n).groupby("enroll")
        test_group = test_cohort_score.groupby("test").head(top_n).groupby("test")

    enroll_mean = enroll_group["score"].mean()
    enroll_std = enroll_group["score"].std()
    test_mean = test_group["score"].mean()
    test_std = test_group["score"].std()

    for _, row in tqdm(input_score.iterrows(), ncols=90, desc='asnorm'):
        enroll_key, test_key, score = row
        
        if cross_select:
            normed_score = 0.5 * ((score - enroll_mean[enroll_key, test_key]) / enroll_std[enroll_key, test_key] + \
                                 (score - test_mean[enroll_key, test_key]) / test_std[enroll_key, test_key])
        else:
            normed_score = 0.5 * ((score - enroll_mean[enroll_key]) / enroll_std[enroll_key] + \
                                (score - test_mean[test_key]) / test_std[test_key])
        
        if output_statistics:
            output_score.append([enroll_key, test_key, enroll_mean[enroll_key], \
                enroll_std[enroll_key], test_mean[test_key], test_std[test_key], normed_score])
        else:
            output_score.append([enroll_key, test_key, normed_score])

    return output_score

def qmf(sv_framework, train_set, eval_input, eval_label, path, len_long=6, len_max=30, num_trials=30000, use_utt_len=False, use_embed_magnitude=False, 
            use_asnorm_metric=False, num_iter=50, converged_loss=1e-4, lr=1e-2):
    # init train set
    train_trials = sample_QMF_train_set(path + '/score_cal/qmf_train_samples.txt', train_set, len_long=len_long, len_max=len_max, num_trials=num_trials)
    
    # enrollment
    items = []
    for item in train_trials:
        items.append(SV_EnrollmentItem(key=item.key1, path=item.key1, speaker=item.key1))
        items.append(SV_EnrollmentItem(key=item.key2, path=item.key2, speaker=item.key2))        
    loader = DataLoader(
        WavLoader(items),
        num_workers=10,
        batch_size=1,
        pin_memory=True,
    )
    embeddings = SV_enrollment(sv_framework, loader, run_on_ddp=False)

    if use_asnorm_metric:
        _, output_scores = adaptive_s_norm(path + '/score_cal', items, sv_framework, train_trials, embeddings, get_score=True)
        cos_sims = []
        mean_enroll_cohort = []
        mean_test_cohort = []
        std_enroll_cohort = []
        std_test_cohort = []
        
        for _, _, mec, sec, mtc, stc, score in output_scores:
            cos_sims.append(score)
            mean_enroll_cohort.append(mec)
            mean_test_cohort.append(mtc)
            std_enroll_cohort.append(sec)
            std_test_cohort.append(stc)
            
    else:
        _, cos_sims = test_SV_EER(train_trials, mono_embedding=embeddings, get_score=True)
    
    qmf_train_set = []
    for i in range(len(cos_sims)):
        input_m = []
        input_m.append(cos_sims[i])
        
        if use_utt_len:
            input_m.append(sf.info(train_trials[i].key1).duration)
            input_m.append(sf.info(train_trials[i].key2).duration)
            
        if use_embed_magnitude:
            e1 = embeddings[train_trials[i].key1]
            e2 = embeddings[train_trials[i].key2]
            input_m.append(torch.sqrt((e1 * e1).sum()) / e1.numel())
            input_m.append(torch.sqrt((e2 * e2).sum()) / e2.numel())
        
        if use_asnorm_metric:
            input_m.append(mean_enroll_cohort[i])
            input_m.append(mean_test_cohort[i])
            input_m.append(std_enroll_cohort[i])
            input_m.append(std_test_cohort[i])
        
        qmf_train_set.append((input_m, train_trials[i].label))
    
    qmf = QMF(len(qmf_train_set[0][0]), num_iter=num_iter, converged_loss=converged_loss, lr=lr)
    qmf.train(qmf_train_set)
    
    scores = []
    for x in eval_input:
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        x = qmf(x).item()
        scores.append(x)
    
    # EER
    eer = calculate_EER(scores, eval_label)
    
    return eer
    
def sample_QMF_train_set(path, dataset, len_long=6, len_max=30, num_trials=30000):
    assert num_trials % 6 == 0
    
    # list -> dict (spk: path)
    short_utts = {}
    long_utts = {}
    
    for item in tqdm(dataset, desc='sample_QMF_train_set', ncols=90):
        duration = sf.info(item.path).duration
        if len_max < duration:
            continue
        elif len_long < duration:
            try:
                long_utts[item.speaker].append(item.path)
            except:
                short_utts[item.speaker] = []
                long_utts[item.speaker] = [item.path]
        else:
            try:
                short_utts[item.speaker].append(item.path)
            except:
                short_utts[item.speaker] = [item.path]
                long_utts[item.speaker] = []
                
    # sampling (same speaker / short-short pair)
    samples = []
    speakers = list(short_utts.keys())
    while len(samples) < num_trials // 6 * 1:
        spk = random.choice(speakers)
        if len(short_utts[spk]) < 2:
            continue
        utt1 = short_utts[spk].pop(random.randint(0, len(short_utts[spk]) - 1))
        utt2 = short_utts[spk].pop(random.randint(0, len(short_utts[spk]) - 1))
        samples.append(SV_Trial(key1=utt1, key2=utt2, label=1))
    
    # sampling (same speaker / short-long pair)
    while len(samples) < num_trials // 6 * 2:
        spk = random.choice(speakers)
        if len(short_utts[spk]) < 1 or len(long_utts[spk]) < 1:
            continue
        utt1 = short_utts[spk].pop(random.randint(0, len(short_utts[spk]) - 1))
        utt2 = long_utts[spk].pop(random.randint(0, len(long_utts[spk]) - 1))
        samples.append(SV_Trial(key1=utt1, key2=utt2, label=1))
    
    # sampling (same speaker / long-long pair)
    while len(samples) < num_trials // 6 * 3:
        spk = random.choice(speakers)
        if len(long_utts[spk]) < 2:
            continue
        utt1 = long_utts[spk].pop(random.randint(0, len(long_utts[spk]) - 1))
        utt2 = long_utts[spk].pop(random.randint(0, len(long_utts[spk]) - 1))
        samples.append(SV_Trial(key1=utt1, key2=utt2, label=1))
     
    # sampling (different speaker / short-short pair)
    while len(samples) < num_trials // 6 * 4:
        spk1 = random.choice(speakers)
        spk2 = random.choice(speakers)
        if len(short_utts[spk1]) < 1 or len(short_utts[spk2]) < 1 or spk1 == spk2:
            continue
        utt1 = short_utts[spk1].pop(random.randint(0, len(short_utts[spk1]) - 1))
        utt2 = short_utts[spk2].pop(random.randint(0, len(short_utts[spk2]) - 1))
        samples.append(SV_Trial(key1=utt1, key2=utt2, label=0))
    
    # sampling (different speaker / short-long pair)
    while len(samples) < num_trials // 6 * 5:
        spk1 = random.choice(speakers)
        spk2 = random.choice(speakers)
        if len(short_utts[spk1]) < 1 or len(long_utts[spk2]) < 1 or spk1 == spk2:
            continue
        utt1 = short_utts[spk1].pop(random.randint(0, len(short_utts[spk1]) - 1))
        utt2 = long_utts[spk2].pop(random.randint(0, len(long_utts[spk2]) - 1))
        samples.append(SV_Trial(key1=utt1, key2=utt2, label=0))
    
    # sampling (different speaker / long-long pair)
    while len(samples) < num_trials // 6 * 6:
        spk1 = random.choice(speakers)
        spk2 = random.choice(speakers)
        if len(long_utts[spk1]) < 1 or len(long_utts[spk2]) < 1 or spk1 == spk2:
            continue
        utt1 = long_utts[spk1].pop(random.randint(0, len(long_utts[spk1]) - 1))
        utt2 = long_utts[spk2].pop(random.randint(0, len(long_utts[spk2]) - 1))
        samples.append(SV_Trial(key1=utt1, key2=utt2, label=0))
    
    # write file
    f = open(path, 'w')
    for item in samples:
        f.write(f'{item.label} {item.key1} {item.key2}\n')
    f.close()
    
    return samples