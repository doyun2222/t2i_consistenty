import torch
import numpy as np
import torch.nn.functional as F

from skimage import filters
from collections import defaultdict


def get_dynamic_threshold(tensor):
    return filters.threshold_otsu(tensor.cpu().numpy())

def attn_map_to_binary(attention_map, scaler=1.):
    attention_map_np = attention_map.cpu().numpy()
    threshold_value = filters.threshold_otsu(attention_map_np) * scaler
    binary_mask = (attention_map_np > threshold_value).astype(np.uint8)

    return binary_mask



def create_token_indices(prompts, batch_size, concept_token, tokenizer):
    if isinstance(concept_token, str):
        concept_token = [concept_token]
    
    concept_token_id = [tokenizer.encode(x, add_special_tokens=False)[0] for x in concept_token]
    
    tokens = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors='pt')['input_ids']

    token_indices = torch.full((len(concept_token), batch_size), -1, dtype=torch.int64)
    for i, token_id in enumerate(concept_token_id):
        batch_loc, token_loc = torch.where(tokens == token_id)
        token_indices[i, batch_loc] = token_loc
    
    return token_indices

class MaskingDropout():
    def __init__(self, attention_store_kwargs):
        self.attn_res = attention_store_kwargs.get('attn_res', (32,32))
        self.token_indices = attention_store_kwargs['token_indices']
        bsz = self.token_indices.size(1)
        self.mask_background_query = attention_store_kwargs.get('mask_background_query', False)
        self.original_attn_masks = attention_store_kwargs.get('original_attn_masks', None)
        self.extended_mapping = attention_store_kwargs.get('extended_mapping', torch.ones(bsz, bsz).bool())
        self.mask_dropout = attention_store_kwargs.get('mask_dropout', 0.0)
        torch.manual_seed(0) # For dropout mask reproducibility

        self.curr_iter = 0
        self.ALL_RES = [32, 64]
        self.step_store = defaultdict(list)
        self.attn_masks = {res: None for res in self.ALL_RES}
        self.last_mask = {res: None for res in self.ALL_RES}
        self.last_mask_dropout = {res: None for res in self.ALL_RES}
        
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str, attn_heads: int):
        if is_cross and attn.shape[1] == np.prod(self.attn_res):
            guidance_attention = attn[attn.size(0)//2:]
            batched_guidance_attention = guidance_attention.reshape([guidance_attention.shape[0]//attn_heads, attn_heads, *guidance_attention.shape[1:]])
            batched_guidance_attention = batched_guidance_attention.mean(dim=1)
            self.step_store[place_in_unet].append(batched_guidance_attention)
            
    def aggregate_last_steps_attention(self) -> torch.Tensor:
        attn_maps = torch.cat([torch.stack(x[-20:]) for x in self.step_store.values()]).mean(dim=0)
        bsz, wh, _ = attn_maps.shape
        
        agg_attn_maps = []
        for i in range(bsz):
            curr_prompt_indices = []
            
            for concept_token_indices in self.token_indices:
                # print(f"concept_token_indices: {concept_token_indices}")
                if concept_token_indices[i] != -1:
                    # print(f"attention_maps[i, :, concept_token_indices[i]]: {attn_maps[i, :, concept_token_indices[i]].shape}")
                    # curr_prompt_indices.append(attn_maps[i, :, concept_token_indices[i]].view(*self.attn_res)])
                    curr_prompt_indices.append(attn_maps[i, :, concept_token_indices[i]].view(*self.attn_res))
                                    
            agg_attn_maps.append(torch.stack(curr_prompt_indices))
            
        for tgt_size in self.ALL_RES:
            pixels = tgt_size ** 2
            tgt_agg_attn_maps = [F.interpolate(x.unsqueeze(1), size=tgt_size, mode='bilinear').squeeze(1) for x in agg_attn_maps]

            attn_masks = []
            for batch_item_map in tgt_agg_attn_maps:
                concept_attn_masks = []

                for concept_maps in batch_item_map:
                    concept_attn_masks.append(torch.from_numpy(attn_map_to_binary(concept_maps, 1.)).to(attn_maps.device).bool().view(-1))

                concept_attn_masks = torch.stack(concept_attn_masks, dim=0).max(dim=0).values
                attn_masks.append(concept_attn_masks)

            attn_masks = torch.stack(attn_masks)
            self.last_mask[tgt_size] = attn_masks.clone()

            # Add mask dropout
            if self.curr_iter < 1000:
                rand_mask = (torch.rand_like(attn_masks.float()) < self.mask_dropout)
                attn_masks[rand_mask] = False

            self.last_mask_dropout[tgt_size] = attn_masks.clone()
        
        
    def get_extended_attn_mask_instance(self, width, i):
        attn_mask = self.last_mask_dropout[width]
        if attn_mask is None:
            return None
        
        n_patches = width**2
        

        output_attn_mask = torch.zeros((attn_mask.shape[0] * attn_mask.shape[1],), device=attn_mask.device, dtype=torch.bool)
        for j in range(attn_mask.shape[0]):
            if i==j:
                output_attn_mask[j*n_patches:(j+1)*n_patches] = 1
            else:
                if self.extended_mapping[i,j]:
                    if not self.mask_background_query:
                        output_attn_mask[j*n_patches:(j+1)*n_patches] = attn_mask[j].unsqueeze(0) #.expand(n_patches, -1)
                    else:
                        raise NotImplementedError('mask_background_query is not supported anymore')
                        output_attn_mask[0, attn_mask[i], k*n_patches:(k+1)*n_patches] = attn_mask[j].unsqueeze(0).expand(attn_mask[i].sum(), -1)

        return output_attn_mask