# Consistent image generation project

## 설명
이 코드는 `dataset/single_object/consistory_prompt_benchmark.yaml` 파일의 prompts를 입력으로 사용하여 이미지를 생성합니다.
- 본 코드는 Stable Diffusion XL를 사용했습니다.
- 현재 baseline 코드는 한 번에 5개 prompt를 입력받아 5개의 결과물을 생성합니다.
- 실행 시, 약 VRAM 18GB를 소모합니다. (A6000 GPU 기준)
  
## 설치 방법
```bash
conda env create --file environment.yml
```

## 실행 방법
```bash
python python main_sdxl.py --output_dir "results_sdxl/test"
```

## 파라미터
- **`--seed`**: 실행 결과를 재현하는데 사용되는 값 (default: 42)
- **`--device`**: 실행할 GPU 번호 지정 (default: 'cuda:0')
- **`--guidance_scale`**: 프롬프트 영향력 지정 (default: 5.0)
- **`--mask_dropout`**: dropout 확률 지정 (default: 0.5)
- **`--pretrained_model`**: 지정한 Stable Diffusion 모델 불러오기 
- **`--output_dir`**: 생성된 이미지 저장 경로 지정
- **`--single_benchmark_dir`**: 벤치마크 데이터셋 경로 지정


