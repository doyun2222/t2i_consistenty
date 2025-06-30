import os
import torch
import yaml
import argparse

from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline
from src.module.sdxl_pipeline import CustomStableDiffusionXLPipeline
from src.module.mask_dropout import create_token_indices

parser = argparse.ArgumentParser()

# Environment
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda:0')

# Inference
parser.add_argument('--guidance_scale', type=float, default=5.0)
parser.add_argument('--num_inference_steps', type=int, default=50)
parser.add_argument('--mask_dropout', type=float, default=0.5)

# Path
parser.add_argument('--pretrained_model', type=str, default='stabilityai/stable-diffusion-xl-base-1.0')
parser.add_argument('--output_dir', type=str, default='results_sdxl/0628_shared_baseline')
parser.add_argument('--single_benchmark_dir', type=str, default='dataset/single_object/consistory_prompt_benchmark.yaml')

args = parser.parse_args()


def single_benchmark(dataset):
    benchmark = {}
    
    for key, values in dataset.items():
        all_prompts = []
        for v in values:
            all_prompts.append((v['prompts'], v['index'], v['concept_token']))
        benchmark[key] = all_prompts  
        
    return benchmark


def main(args):
    # single benchmark load
    with open(args.single_benchmark_dir, 'r') as f:
        dataset = yaml.safe_load(f)
    
    s_bench = single_benchmark(dataset)
    
    # SD v2.1-base
    # pipeline = StableDiffusionXLPipeline.from_pretrained(
    #     args.pretrained_model,
    #     torch_dtype=torch.float16,
    #     variant="fp16",
    #     use_safetensors=True,
    # ).to(args.device)
    
    custom_pipeline = CustomStableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(args.device)
    
    
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    
    # FreeU
    custom_pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    
    # Training-free approach
    total = sum(len(prompts) for prompts in s_bench.values())
    with tqdm(total=total, desc="Image Generating") as pbar:
        tokenizer = custom_pipeline.tokenizer
        
        for k, prompts in s_bench.items():
            for i, (prompt, index, concept_token) in enumerate(prompts):
                
                batch_size = len(prompt)
                token_indices = create_token_indices(prompt, batch_size, concept_token, tokenizer)
                
                attention_store_kwargs = {
                    'token_indices': token_indices,
                    'mask_dropout': args.mask_dropout,
                }
                
                result = custom_pipeline(prompt, generator=generator,
                                         num_inference_steps=args.num_inference_steps, 
                                         guidance_scale=args.guidance_scale,
                                         attention_store_kwargs=attention_store_kwargs,
                                  )
                
                output_subdir = os.path.join(args.output_dir, f"{k}/{index}")
                os.makedirs(output_subdir, exist_ok=True)
                
                for j, image in enumerate(result.images):
                    image_path = os.path.join(output_subdir, f"img_{j}.png")
                    image.save(image_path)

                pbar.update(1)  # 한 샘플 처리 후 1씩 증가
                
                
if __name__ == '__main__':
    main(args)