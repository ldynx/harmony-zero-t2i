# Code Appendix

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