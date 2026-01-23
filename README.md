# Harmonizing Visual and Textual Embeddings for Zero-Shot Text-to-Image Customization (AAAI 2025)

This repository contains the official implementation of 
[<U>Harmonizing Visual and Textual Embeddings for Zero-Shot Text-to-Image Customization</U>](https://arxiv.org/pdf/2403.14155).

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://ldynx.github.io/harmony-zero-t2i//)
[![arXiv 2312.02503](https://img.shields.io/badge/arXiv-2312.02503-red)](https://arxiv.org/abs/2403.14155)

### Run BLIP-Diffusion
1. Run the following command: <br/>
```python
python run_blip_diffusion.py \
    --data_dir='./example/01.jpg' \
    --data_class='cat' \
    --prompt='jumping' \
    --output_dir='./outputs'
```


### Run ELITE
1. Download gloabal and local mappers from [ELITE homepage](https://github.com/csyxwei/ELITE/tree/main).
2. Run basline or ours (run_elite_ours.py) <br/>
```python
python run_elite.py \
    --global_mapper_path="path-to-global_mapper.pt" \
    --local_mapper_path="path-to-local_mapper.pt" \
    --test_data_dir='./example/01.jpg' \ 
    --template='a * riding a bike' \
    --output_dir='./outputs' \
```

### Computing Infrastructure
- GPU Model : TITAN RTX
- Memory : 13GB (BLIP-Diffusion) / 9GB (ELITE)
- Operating system: Ubuntu 18.04.5 LTS
