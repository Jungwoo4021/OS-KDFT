from typing import Optional, Tuple, Union
 
import math
import random
import numpy as np
import torch
import torch.nn as nn
import transformers
import transformers.models.hubert.modeling_hubert as hubert
from transformers import AutoConfig, HubertConfig, HubertModel
from transformers.modeling_outputs import BaseModelOutput

__all__ = ['StudentHubert_DistHubt', 'StudentHubert_FitHubt']

BASE_960 = 'facebook/hubert-base-ls960'


############
## Models ##
############
class StudentHubert_DistHubt(nn.Module):
    def __init__(self, num_hidden_layer, hidden_size, return_all_hiddens=True, init_teacher_param=None, os_kdft_adapter=None):
        super(StudentHubert_DistHubt, self).__init__()
        self.return_all_hiddens = return_all_hiddens

        # set transformer encoder
        config = AutoConfig.from_pretrained(BASE_960)
        config.num_hidden_layers = num_hidden_layer
        config.hidden_size = hidden_size
        self.hubert = CustomHubertModel(config, os_kdft_adapter=os_kdft_adapter)

        # weight initialization
        teacher = HubertModel.from_pretrained(
            BASE_960,
            from_tf=bool(".ckpt" in BASE_960),
            config=AutoConfig.from_pretrained(BASE_960),
            revision="main",
            ignore_mismatched_sizes=False,
        )
        self.hubert.feature_extractor.load_state_dict(teacher.feature_extractor.state_dict(), strict=False)
        self.hubert.feature_projection.load_state_dict(teacher.feature_projection.state_dict(), strict=False)
        if init_teacher_param is not None:
            for i in range(num_hidden_layer):
                self.hubert.encoder.layers[i].load_state_dict(teacher.encoder.layers[init_teacher_param[i]].state_dict(), strict=False)
        
    def forward(self, x, idx_without_adapter=None):
        x = self.hubert(x, output_hidden_states=self.return_all_hiddens, idx_without_adapter=idx_without_adapter)
        
        if self.return_all_hiddens:
            return torch.stack(x.hidden_states, dim=1)
        else:
            return x.last_hidden_state

class StudentHubert_FitHubt(nn.Module):
    def __init__(self, num_hidden_layer, hidden_size_s, hidden_size_t, sequence_length, return_all_hiddens=True, os_kdft_adapter=None):
        super(StudentHubert_FitHubt, self).__init__()
        self.return_all_hiddens = return_all_hiddens

        # set transformer encoder
        config = HubertConfig.from_pretrained(BASE_960)
        config.num_hidden_layers = num_hidden_layer
        config.hidden_size = hidden_size_s
        config.num_feat_extract_layers = 9
        config.conv_dim = (128, 256, 256, 256, 256, 256, 512, 512, 512)
        config.conv_stride = (5, 1, 2, 2, 2, 2, 1, 2, 2)
        config.conv_kernel =  (10, 1, 3, 3, 3, 3, 1, 2, 2)
        config.intermediate_size = hidden_size_s
        config.do_stable_layer_norm = True
        self.hubert = CustomHubertModel(config, use_time_reduction=True, os_kdft_adapter=os_kdft_adapter)

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
            
class CustomHubertModel(hubert.HubertModel):
    def __init__(self, config, use_time_reduction=False, os_kdft_adapter=None):
        super().__init__(config)

        self.config = config
        self.feature_extractor = hubert.HubertFeatureEncoder(config)
        self.feature_projection = hubert.HubertFeatureProjection(config)
        self.use_time_reduction = use_time_reduction
        if use_time_reduction:
            self.time_reduction_layer = nn.Conv1d(config.conv_dim[-1], config.conv_dim[-1], kernel_size=2, stride=2, bias=False)
        
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())
        
        if config.do_stable_layer_norm:
            self.encoder = CustomHubertEncoderStableLayerNorm(config, os_kdft_adapter)
        else:
            self.encoder = CustomHubertEncoder(config, os_kdft_adapter)

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
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            idx_without_adapter=idx_without_adapter,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        
class CustomHubertEncoder(nn.Module):
    def __init__(self, config, os_kdft_adapter=None):
        super().__init__()
        self.config = config
        self.pos_conv_embed = hubert.HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([CustomHubertEncoderLayer(config, os_kdft_adapter=os_kdft_adapter) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
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
            dropout_probability = np.random.uniform(0, 1)

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
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, idx_without_adapter=idx_without_adapter
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

class CustomHubertEncoderStableLayerNorm(nn.Module):
    def __init__(self, config, os_kdft_adapter=None):
        super().__init__()
        self.config = config
        self.pos_conv_embed = hubert.HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [CustomHubertEncoderLayerStableLayerNorm(config, os_kdft_adapter=os_kdft_adapter) for _ in range(config.num_hidden_layers)]
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
            # make sure padded tokens are not attended to
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
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = transformers.deepspeed.is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
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
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, idx_without_adapter=idx_without_adapter
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        
class CustomHubertEncoderLayer(nn.Module):
    def __init__(self, config, os_kdft_adapter=None):
        super().__init__()
        self.attention = hubert.HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = hubert.HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # FT adapter
        self.use_ft_adapter = os_kdft_adapter is not None
        if self.use_ft_adapter:
            self.os_kdft_adapter = OS_KDFT_Adapter(config.hidden_size, os_kdft_adapter)
            
    def forward(self, hidden_states, attention_mask=None, output_attentions=False, idx_without_adapter=None):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        
        # FT adapter
        if self.use_ft_adapter:
            if idx_without_adapter is None:
                h = self.os_kdft_adapter(hidden_states)    
                hidden_states = hidden_states + self.feed_forward(hidden_states) + h
            else:
                # separate branch (KD & SV)
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
        else:
            hidden_states = hidden_states + self.feed_forward(hidden_states)
        
        hidden_states = self.final_layer_norm(hidden_states)
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class CustomHubertEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config, os_kdft_adapter=None):
        super().__init__()
        self.attention = hubert.HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = hubert.HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # FT adapter
        self.use_ft_adapter = os_kdft_adapter is not None
        if self.use_ft_adapter:
            self.os_kdft_adapter = OS_KDFT_Adapter(config.hidden_size, os_kdft_adapter)
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        idx_without_adapter = None
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        
        # FT adapter
        if self.use_ft_adapter:
            if idx_without_adapter is None:
                h = self.os_kdft_adapter(hidden_states)    
                hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states)) + h
            else:
                # separate branch (KD & SV)
                ff_hidden_states = self.feed_forward(self.final_layer_norm(hidden_states))
                kd_hidden_states, ft_hidden_states = hidden_states[:idx_without_adapter, :, :], hidden_states[idx_without_adapter:, :, :]
                kd_ff_hidden_states, ft_ff_hidden_states = ff_hidden_states[:idx_without_adapter, :, :], ff_hidden_states[idx_without_adapter:, :, :]

                # KD branch
                kd_hidden_states = kd_hidden_states + kd_ff_hidden_states
                
                # FT branch
                h = self.os_kdft_adapter(ft_hidden_states)
                ft_hidden_states = ft_hidden_states + ft_ff_hidden_states + h

                # merge
                hidden_states = torch.cat((kd_hidden_states, ft_hidden_states), dim=0)
        else:
            hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states,)

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
