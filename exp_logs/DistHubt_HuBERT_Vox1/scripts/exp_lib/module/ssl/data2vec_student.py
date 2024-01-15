import math
from typing import Optional, Tuple, Union
import numpy as np
import random

import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig, Data2VecAudioConfig, Data2VecAudioModel
from transformers.modeling_outputs import BaseModelOutput
import transformers.models.data2vec.modeling_data2vec_audio as d2v

BASE960 = 'facebook/data2vec-audio-base-960h'

############
## Models ##
############
class StudentData2Vec_DistHubt(nn.Module):
    def __init__(self, num_hidden_layer, hidden_size, return_all_hiddens=True, init_teacher_param=None, os_kdft_adapter=None):
        super(StudentData2Vec_DistHubt, self).__init__()
        self.return_all_hiddens = return_all_hiddens

        # set transformer encoder
        config = AutoConfig.from_pretrained(BASE960)
        config.num_hidden_layers = num_hidden_layer
        config.hidden_size = hidden_size
        self.data2vec = CustomData2VecAudioModel(config, os_kdft_adapter_hidden=os_kdft_adapter)

        # weight initialization
        teacher = Data2VecAudioModel.from_pretrained(
            BASE960,
            from_tf=bool(".ckpt" in BASE960),
            config=AutoConfig.from_pretrained(BASE960),
            revision="main",
            ignore_mismatched_sizes=False,
        )
        self.data2vec.feature_extractor.load_state_dict(teacher.feature_extractor.state_dict(), strict=False)
        self.data2vec.feature_projection.load_state_dict(teacher.feature_projection.state_dict(), strict=False)
        if init_teacher_param is not None:
            for i in range(num_hidden_layer):
                self.data2vec.encoder.layers[i].load_state_dict(teacher.encoder.layers[init_teacher_param[i]].state_dict(), strict=False)
        
    def forward(self, x, idx_without_adapter=None):
        x = self.data2vec(x, output_hidden_states=self.return_all_hiddens, idx_without_adapter=idx_without_adapter)
        
        if self.return_all_hiddens:
            return torch.stack(x.hidden_states, dim=1)
        else:
            return x.last_hidden_state

class StudentHubert_FitHubt(nn.Module):
    def __init__(self, num_hidden_layer, hidden_size_s, hidden_size_t, sequence_length, return_all_hiddens=True, os_kdft_adapter=None):
        super(StudentHubert_FitHubt, self).__init__()
        self.return_all_hiddens = return_all_hiddens

        # set transformer encoder
        config = Data2VecAudioConfig.from_pretrained(BASE960)
        config.num_hidden_layers = num_hidden_layer
        config.hidden_size = hidden_size_s
        config.num_feat_extract_layers = 9
        config.conv_dim = (128, 256, 256, 256, 256, 256, 512, 512, 512)
        config.conv_stride = (5, 1, 2, 2, 2, 2, 1, 2, 2)
        config.conv_kernel =  (10, 1, 3, 3, 3, 3, 1, 2, 2)
        config.intermediate_size = hidden_size_s
        config.do_stable_layer_norm = True
        self.hubert = CustomData2VecAudioModel(config, use_time_reduction=True, os_kdft_adapter_hidden=os_kdft_adapter)

        self.hidden_size_s = hidden_size_s
        self.hidden_size_t = hidden_size_t
        self.heads = FitHubtHead(hidden_size_s, hidden_size_t, num_hidden_layer, sequence_length)
        
    def forward(self, x, idx_without_adapter=None):
        x = self.hubert(x, output_hidden_states=self.return_all_hiddens, idx_without_adapter=idx_without_adapter)
        
        if self.return_all_hiddens:
            x = torch.stack(x.hidden_states, dim=1)
            x = self.heads(x)
            return x
        else:
            raise NotImplementedError('FitHuBERT does not support single hidden return')



#################
## Sub modules ##
#################
class FitHubtHead(nn.Module):
    def __init__(self, hidden_size_s, hidden_size_t, num_layer, num_frames):
        super().__init__()
        self.tconv = nn.ModuleList([nn.ConvTranspose1d(hidden_size_s, hidden_size_s, kernel_size=2, stride=2) for _ in range(num_layer)])
        self.fc = nn.ModuleList([nn.Linear(hidden_size_s, hidden_size_t) for _ in range(num_layer)])
        self.num_frames = num_frames
    
    def forward(self, x):
        hs = []
        batch, layer, _, hidden = x.size()
            
        for i in range(1, layer):
            _x = x[:, i, :, :]
            _x = _x.permute(0, 2, 1) # batch, hidden, time        
            s = (batch, hidden, self.num_frames)
            _x = self.tconv[i - 1](_x, output_size=s)
            
            _x = _x.permute(0, 2, 1) # batch, time, hidden
            _x = self.fc[i - 1](_x)
            
            hs.append(_x)
        
        hs = torch.stack(hs, dim=1)
        
        return hs
            
class CustomData2VecAudioModel(Data2VecAudioModel):
    def __init__(self, config, use_time_reduction=False, os_kdft_adapter_hidden=None):
        super().__init__(config)
        self.config = config
        self.feature_extractor = d2v.Data2VecAudioFeatureEncoder(config)
        self.feature_projection = d2v.Data2VecAudioFeatureProjection(config)
        
        self.use_time_reduction = use_time_reduction
        if use_time_reduction:
            self.time_reduction_layer = nn.Conv1d(config.conv_dim[-1], config.conv_dim[-1], kernel_size=2, stride=2, bias=False)
        
        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = CustomData2VecAudioEncoder(config, os_kdft_adapter_hidden=os_kdft_adapter_hidden)

        self.adapter = d2v.Data2VecAudioAdapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        idx_without_adapter: Optional[list] = None,
    ) -> Union[Tuple, transformers.modeling_outputs.BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # CNN
        extract_features = self.feature_extractor(input_values)
        
        # time reduction layer (for fithubert)
        if self.use_time_reduction:
            extract_features = self.time_reduction_layer(extract_features)
        
        extract_features = extract_features.transpose(1, 2)
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask, add_apapter=False)

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            idx_without_adapter=idx_without_adapter,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
     
class CustomData2VecAudioEncoder(nn.Module):
    def __init__(self, config, os_kdft_adapter_hidden=None):
        super().__init__()
        self.config = config
        self.pos_conv_embed = d2v.Data2VecAudioPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [CustomData2VecAudioEncoderLayer(config, os_kdft_adapter_hidden=os_kdft_adapter_hidden) for i in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        idx_without_adapter=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = transformers.deepspeed.is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        idx_without_adapter=idx_without_adapter
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class CustomData2VecAudioEncoderLayer(nn.Module):
    def __init__(self, config, os_kdft_adapter_hidden=None):
        super().__init__()
        self.attention = d2v.Data2VecAudioAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = d2v.Data2VecAudioFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # FT adapter
        self.use_ft_adapter = os_kdft_adapter_hidden is not None
        if self.use_ft_adapter:
            self.os_kdft_adapter = OS_KDFT_Adapter(config.hidden_size, os_kdft_adapter_hidden)
            
    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0, idx_without_adapter=None):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        # FT adapter
        if self.use_ft_adapter:
            if idx_without_adapter is None:
                hidden_states = self.layer_norm(hidden_states)
                h = self.os_kdft_adapter(hidden_states)    
                hidden_states = hidden_states + self.feed_forward(hidden_states) + h
                hidden_states = self.final_layer_norm(hidden_states)
            else:
                # separate branch (KD & SV)
                hidden_states = self.layer_norm(hidden_states)
                ff_hidden_states = self.feed_forward(hidden_states)
                kd_hidden_states, ft_hidden_states = hidden_states[:idx_without_adapter, :, :], hidden_states[idx_without_adapter:, :, :]
                kd_ff_hidden_states, ft_ff_hidden_states = ff_hidden_states[:idx_without_adapter, :, :], ff_hidden_states[idx_without_adapter:, :, :]

                # KD branch
                kd_hidden_states = kd_hidden_states + kd_ff_hidden_states
                
                # FT branch
                h = self.os_kdft_adapter(ft_hidden_states)
                ft_hidden_states = ft_hidden_states + ft_ff_hidden_states + h
                
                # merge
                hidden_states = torch.cat((kd_hidden_states, ft_hidden_states), dim=0)
                hidden_states = self.final_layer_norm(hidden_states)
        else:
            hidden_states = self.layer_norm(hidden_states)
            hidden_states = hidden_states + self.feed_forward(hidden_states)
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, position_bias)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class OS_KDFT_Adapter(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.down = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.up = nn.Linear(hidden_channels, in_channels)
        
    def forward(self, x):
        # projection
        x = self.down(x)
        x = self.relu(x)
        if self.training:
            x = x + torch.normal(mean=0, std=random.uniform((x.abs().mean() * 0.1).item(), (x.abs().mean() * 0.5).item()), size=x.size(), device=x.device, dtype=x.dtype)
        x = self.up(x)
        
        return x