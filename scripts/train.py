import copy
from tqdm import tqdm
import torch 
import torch.nn.functional as F
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from ddp_util import all_gather, synchronize
import torch.distributed as dist

class ModelTrainer:
    args = None
    vox = None
    wav2vec_t = None
    wav2vec_s = None
    ecapa = None
    logger = None
    criterion = None
    optimizer = None
    lr_scheduler = None
    lr_guide = None
    train_set = None
    train_sampler = None
    train_loader = None
    test_loader_O = None
    test_loader_E = None
    loss_KD = None
    
    def run(self):
        self.best_eer = 100
        self.step = 0

        for epoch in range(1, self.args['epoch'] + 1):
            self.lr_decay(epoch)
            self.train(epoch)
            self.test(epoch, epoch == self.args['epoch'])


    def train(self, epoch):
        self.wav2vec_t.eval()
        self.wav2vec_s.train()
        self.ecapa.train()
        self.train_sampler.set_epoch(epoch)

        count = 0
        loss_spk_sum = 0
        loss_kd_sum = 0
        loss_total = 0
        
        with tqdm(total=len(self.train_loader), ncols=90) as pbar:
            kd_weight = (self.args['epoch'] - epoch) / self.args['epoch']
            for x1, x2, labels in self.train_loader:
                # clear grad
                self.optimizer.zero_grad()

                # feed forward - teacher
                x1 = x1.to(torch.float32).to(self.args['device'])
                x2 = x2.to(torch.float32).to(self.args['device'])
                labels = labels.to(self.args['device'])
                
                with torch.set_grad_enabled(False):   
                    x_t = self.wav2vec_t(x1).last_hidden_state

                # feed forward - student
                x_s, x_s_adapter = self.wav2vec_s(x2.repeat(2, 1), flag_train=True)
                x_s = x_s.last_hidden_state
                x_s_adapter = x_s_adapter.last_hidden_state

                loss_kd = self.loss_KD(x_s, x_t) * 100 * kd_weight

                loss_spk = self.ecapa(x_s_adapter.permute(0, 2, 1), labels)

                loss = loss_kd + loss_spk

                # backpropagation
                loss.backward()
                self.optimizer.step()

                # log
                if self.logger is not None:
                    count += 1
                    loss_spk_sum += loss_spk.item()
                    loss_kd_sum += loss_kd.item()
                    loss_total += loss.item()
                    if len(self.train_loader) * 0.01 <= count:
                        self.logger.log_metric('Loss/spk', loss_spk_sum / count)
                        self.logger.log_metric('Loss/KD', loss_kd_sum / count)
                        self.logger.log_metric('Loss/total', loss_total / count)
                        count = 0
                        loss_spk_sum = 0
                        loss_kd_sum = 0
                        loss_total = 0
                    
                    # pbar
                    desc = f'{self.args["name"]}-[{epoch}/{self.args["epoch"]}]|(loss): {loss.item():.3f}'
                    pbar.set_description(desc)
                    pbar.update(1)

    def lr_decay(self, step):
        for p_group in self.lr_guide.param_groups:
            sv_lr = p_group['lr']
        
        if step < 10:
            self.kd_lr = sv_lr * (step / 10)
        else:
            self.kd_lr = self.kd_lr * 0.93
            
        kd_params = []
        sv_params = []
        adapter_params = []
        
        for name, param in self.wav2vec_s.named_parameters():
            if 'sv_adapter' in name:
                adapter_params.append(param)
            else:
                kd_params.append(param)
        
        for name, param in self.ecapa.named_parameters():
            sv_params.append(param)

        # optimizer
        self.optimizer = torch.optim.Adam(
            [
                {'params': kd_params, 'lr': self.kd_lr},
                {'params': sv_params, 'lr': sv_lr},
                {'params': adapter_params, 'lr': sv_lr * 10}
            ]
        )

        if self.logger is not None:
            self.logger.log_metric('sv_lr', sv_lr, step=step)
            self.logger.log_metric('kd_lr', self.kd_lr, step=step)
            
        self.lr_scheduler.step()
        
    def test(self, epoch, flag_in_depth_test):
        flag_save_model = [0]
        embeddings_full, embeddings_seg = self.enrollment(self.test_loader_O)

        if self.args['flag_parent']:
            eer = self.calculate_EER(embeddings_full, embeddings_seg, self.vox.trials_O)

            self.logger.log_metric('EER', eer, step=epoch)
            if eer < self.best_eer:
                self.best_eer = eer
                self.logger.log_metric('EER_Best', eer, step=epoch)
                self.logger.save_model(f'W2V2_{self.best_eer:.3f}', self.wav2vec_s.state_dict())
                self.logger.save_model(f'ECAPA_{self.best_eer:.3f}', self.ecapa.state_dict())
                flag_save_model.append(1)
        self._synchronize()

        flag_save_model = all_gather(flag_save_model)
        if sum(flag_save_model):
            self.best_w2v2 = copy.deepcopy(self.wav2vec_s.state_dict())
            self.best_ecapa = copy.deepcopy(self.ecapa.state_dict())

        if flag_in_depth_test:
            self.wav2vec_s.load_state_dict(self.best_w2v2)
            self.ecapa.load_state_dict(self.best_ecapa)
            embeddings_full2, embeddings_seg2 = self.enrollment(self.test_loader_E)
            embeddings_full.update(embeddings_full2)
            embeddings_seg.update(embeddings_seg2)

            if self.args['flag_parent']:
                eer = self.calculate_EER(embeddings_full, embeddings_seg, self.vox.trials_E)
                self.logger.log_metric('EER_E', eer, step=epoch)
                eer = self.calculate_EER(embeddings_full, embeddings_seg, self.vox.trials_H)
                self.logger.log_metric('EER_H', eer, step=epoch)

        self._synchronize()

    def enrollment(self, data_loader):
        self.wav2vec_s.eval()
        self.ecapa.eval()

        keys = []
        embeddings_full = []
        embeddings_seg = []

        with tqdm(total=len(data_loader), ncols=90) as pbar, torch.set_grad_enabled(False):
            for x_full, x_seg, key in data_loader:
                x_full = x_full.to(torch.float32).to(self.args['device'], non_blocking=True)
                x_seg = x_seg.to(torch.float32).to(self.args['device'], non_blocking=True).view(self.args['num_seg'], x_seg.size(-1)) 
                
                x_full = self.wav2vec_s(x_full, flag_train=False).last_hidden_state
                x_full = x_full.permute(0, 2, 1)
                
                x_seg = self.wav2vec_s(x_seg, flag_train=False).last_hidden_state
                x_seg = x_seg.permute(0, 2, 1)
                
                x_full = self.ecapa(x_full).to('cpu')
                x_seg = self.ecapa(x_seg).to('cpu')

                keys.append(key[0])
                embeddings_full.append(x_full[0])
                embeddings_seg.append(x_seg)

                if self.args['flag_parent']:
                    pbar.update(1)

        # synchronize
        self._synchronize()

        keys = all_gather(keys)
        embeddings_full = all_gather(embeddings_full)
        embeddings_seg = all_gather(embeddings_seg)

        full_dict = {}
        seg_dict = {}
        for i in range(len(keys)):
            full_dict[keys[i]] = embeddings_full[i]
            seg_dict[keys[i]] = embeddings_seg[i]

        return full_dict, seg_dict

    def calculate_EER(self, embeddings_full, embeddings_seg, trials):
        labels = []
        cos_sims_full = [[], []]
        cos_sims_seg = [[], []]

        for item in trials:
            cos_sims_full[0].append(embeddings_full[item.key1])
            cos_sims_full[1].append(embeddings_full[item.key2])

            cos_sims_seg[0].append(embeddings_seg[item.key1])
            cos_sims_seg[1].append(embeddings_seg[item.key2])

            labels.append(item.label)

        # cosine_similarity - full
        buffer1 = torch.stack(cos_sims_full[0], dim=0)
        buffer2 = torch.stack(cos_sims_full[1], dim=0)
        cos_sims_full = F.cosine_similarity(buffer1, buffer2)

        # cosine_similarity - seg
        batch = len(labels)
        num_seg = self.args['num_seg']
        buffer1 = torch.stack(cos_sims_seg[0], dim=0).view(batch, num_seg, -1)
        buffer2 = torch.stack(cos_sims_seg[1], dim=0).view(batch, num_seg, -1)
        buffer1 = buffer1.repeat(1, num_seg, 1).view(batch * num_seg * num_seg, -1)
        buffer2 = buffer2.repeat(1, 1, num_seg).view(batch * num_seg * num_seg, -1)
        cos_sims_seg = F.cosine_similarity(buffer1, buffer2)
        cos_sims_seg = cos_sims_seg.view(batch, num_seg * num_seg)
        cos_sims_seg = cos_sims_seg.mean(dim=1)

        cos_sims = (cos_sims_full + cos_sims_seg) * 0.5

        fpr, tpr, _ = metrics.roc_curve(labels, cos_sims, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return eer * 100

    def _synchronize(self):
        torch.cuda.empty_cache()
        dist.barrier()