import torch
import xformers

from typing import Callable, List, Optional, Tuple, Union
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.attention_processor import Attention

class CustomAttnProcessor:
    def __init__(self, place_in_unet, attnstore):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, record_attention=True, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # only need to store attention maps during the Attend and Excite process
        # if attention_probs.requires_grad:
        if record_attention:
            self.attnstore(attention_probs, is_cross, self.place_in_unet, attn.heads)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CustomSharedAttnXFormersAttnProcessor:
    def __init__(self, place_in_unet, attnstore):
        self.place_in_unet = place_in_unet
        self.attnstore = attnstore
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states
        
        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, wh, channel = hidden_states.shape
            height = width = int(wh ** 0.5)
            
        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query).contiguous()
        
        
        ex_out = torch.empty_like(query)

        for i in range(batch_size):
            start_idx = i * attn.heads
            end_idx = start_idx + attn.heads

            attention_mask = self.attnstore.get_extended_attn_mask_instance(width, i%(batch_size//2))

            curr_q = query[start_idx:end_idx]

            if i < batch_size//2:
                curr_k = key[:batch_size//2]
                curr_v = value[:batch_size//2]
            else:
                curr_k = key[batch_size//2:]
                curr_v = value[batch_size//2:]
            
            # if attention_mask == None:
            curr_k = curr_k.flatten(0,1).unsqueeze(0)
            curr_v = curr_v.flatten(0,1).unsqueeze(0)
            # else:
            #     curr_k = curr_k.flatten(0,1)[attention_mask].unsqueeze(0)
            #     curr_v = curr_v.flatten(0,1)[attention_mask].unsqueeze(0)

            curr_k = attn.head_to_batch_dim(curr_k).contiguous()            
            curr_v = attn.head_to_batch_dim(curr_v).contiguous()

            hidden_states = xformers.ops.memory_efficient_attention(
                curr_q, curr_k, curr_v, scale=attn.scale
                # op=self.attention_op, scale=attn.scale
            )

            ex_out[start_idx:end_idx] = hidden_states

        hidden_states = ex_out
        
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
        
    
def register_shared_attn(unet, attnstore):
    attn_procs = {}
    for i, name in enumerate(unet.attn_processors.keys()):
        if name.startswith("mid_block"):
            place_in_unet = f"mid_{i}"
        elif name.startswith("up_blocks"):
            place_in_unet = f"up_{i}"
        elif name.startswith("down_blocks"):
            place_in_unet = f"down_{i}"
        else:
            continue
        
        if name.endswith("attn1.processor"):
            if name.startswith("up_blocks"):
                attn_procs[name] = CustomSharedAttnXFormersAttnProcessor(place_in_unet, attnstore)
            else:    
                # print(f"hi")
                attn_procs[name] = CustomAttnProcessor(place_in_unet, attnstore)    
        else:
            attn_procs[name] = CustomAttnProcessor(place_in_unet, attnstore)
    
    unet.set_attn_processor(attn_procs)
    