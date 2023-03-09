import math
import torch
import torch.nn as nn
import numpy as np
import transformers
import warnings
from typing import Optional, Tuple, Union
import transformers.models.wav2vec2.modeling_wav2vec2 as w2v2

args = None

class Wav2Vec2_Mini(nn.Module):
    def __init__(self, config: transformers.Wav2Vec2Config):  # : type 설정
        super().__init__()
        self.config = config
        self.feature_extractor = w2v2.Wav2Vec2FeatureEncoder(config)
        self.feature_projection = w2v2.Wav2Vec2FeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())
        
        self.encoder = Wav2Vec2EncoderStableLayerNorm(config)

        self.adapter = w2v2.Wav2Vec2Adapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.apply(self._init_weights)  # post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        # gumbel softmax requires special init
        if isinstance(module, w2v2.Wav2Vec2GumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        elif isinstance(module, w2v2.Wav2Vec2PositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, w2v2.Wav2Vec2FeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = w2v2._compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = w2v2._compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        flag_train: Optional[bool],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, transformers.modeling_outputs.Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        if flag_train:
            encoder_outputs, encoder_outputs_adapter = self.encoder(
                hidden_states,
                flag_train=flag_train,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            
            hidden_states_no_adapter = encoder_outputs[0]
            hidden_states = encoder_outputs_adapter[0]

            if self.adapter is not None:
                hidden_states_no_adapter = self.adapter(hidden_states_no_adapter)
                hidden_states = self.adapter(hidden_states)

            output1 = transformers.modeling_outputs.Wav2Vec2BaseModelOutput(
                last_hidden_state=hidden_states_no_adapter,
                extract_features=extract_features,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
            output2 = transformers.modeling_outputs.Wav2Vec2BaseModelOutput(
                last_hidden_state=hidden_states,
                extract_features=extract_features,
                hidden_states=encoder_outputs_adapter.hidden_states,
                attentions=encoder_outputs_adapter.attentions,
            )
            return output1, output2
        else:
            encoder_outputs = self.encoder(
                hidden_states,
                flag_train=flag_train,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            hidden_states = encoder_outputs[0]

            if self.adapter is not None:
                hidden_states = self.adapter(hidden_states)

            return transformers.modeling_outputs.Wav2Vec2BaseModelOutput(
                last_hidden_state=hidden_states,
                extract_features=extract_features,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

class Wav2Vec2EncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = w2v2.Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        flag_train=False,
    ):
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

        # encoder layers without adapter
        if flag_train:
            hidden_states, all_hidden_states, all_self_attentions = self.forward_encoder_layers(
                hidden_states, attention_mask, flag_train=flag_train, output_hidden_states=output_hidden_states, output_attentions=output_attentions
            )
            half_batch = hidden_states.size(0) // 2
            hidden_states, hidden_states_adapter = hidden_states[:half_batch, :, :], hidden_states[half_batch:, :, :]
         
            if all_hidden_states is not None:
                all_hidden_states, all_hidden_states_adapter = all_hidden_states[:half_batch, :, :], all_hidden_states[half_batch:, :, :]
            else:
                all_hidden_states_adapter = None

            if all_self_attentions is not None:
                all_self_attentions, all_self_attentions_adapter = all_self_attentions[:half_batch, :, :], all_self_attentions[half_batch:, :, :]
            else:
                all_self_attentions_adapter = None
                
            hidden_states = self.layer_norm(hidden_states)
            hidden_states_adapter = self.layer_norm(hidden_states_adapter)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                all_hidden_states_adapter = all_hidden_states_adapter + (hidden_states_adapter,)

            output1 = transformers.modeling_outputs.BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
            output2 = transformers.modeling_outputs.BaseModelOutput(
                last_hidden_state=hidden_states_adapter,
                hidden_states=all_hidden_states_adapter,
                attentions=all_self_attentions_adapter,
            )

            return output1, output2

        else:
            hidden_states, all_hidden_states, all_self_attentions = self.forward_encoder_layers(
                hidden_states, attention_mask, flag_train=True, output_hidden_states=output_hidden_states, output_attentions=output_attentions
            )

            hidden_states = self.layer_norm(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            return transformers.modeling_outputs.BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
    
    def forward_encoder_layers(self, hidden_states, attention_mask, flag_train, output_hidden_states, output_attentions):
        deepspeed_zero3_is_enabled = transformers.deepspeed.is_deepspeed_zero3_enabled()

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

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
                        hidden_states, flag_train=flag_train, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        
        return hidden_states, all_hidden_states, all_self_attentions


class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = w2v2.Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = w2v2.Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.sv_adapter = Adapter(in_channels=config.hidden_size, hidden_channels=64, adapter_layernorm_option='in', adapter_scalar='0.1')

    def forward(self, hidden_states: torch.Tensor, flag_train: bool, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        # adapter
        if flag_train:
            half_batch = hidden_states.size(0) // 2
            no_adapt, adapt_h = hidden_states[:half_batch, :, :], hidden_states[half_batch:, :, :]
            no_adapt = no_adapt * 0
            adapt_h = self.sv_adapter(adapt_h)
            identity = torch.cat((no_adapt, adapt_h), dim=0)
            hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states)) + identity
        else:
            hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states)) + self.sv_adapter(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class Adapter(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 hidden_channels=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="none"
                 ):
        super().__init__()
        self.n_embd = in_channels
        self.down_size = hidden_channels

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        
        down = self.down_proj(x)
        
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        
        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output