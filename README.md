# Progressive Growing of GANs

- Reference Paper："[Progressive growing of gans for improved quality, stability, and variation](https://arxiv.org/abs/1710.10196)" 



## Dependencies

- Pytorch >= 1.0 
- [Apex AMP](https://github.com/NVIDIA/apex.git)
- packages in [requirements](requirements.txt)




### 1. Dataset preparation（images of different resolution）
```bash
python src/data_tools/generate_dataset.py --source_path=fitsdata/train --target_path=fitsdata/cache/train
```

### 2. Train 

```bash
python src/train.py models/default/config.ym
```
- **The results during training process are saved in`models`，among which`models\default\generated_data`saves the images during the process
- **`models\default\checkpoints`saves models during the process

### 3. Test

```bash
python src/demo.py -c=models/default/config.yml -m=fitsdata/checkpoint/256x256.ckpt -n=32
```
- **Could change the number of generated pictures
- **The result is in `fitsdata\results`**
- **Use`fitsdata/checkpoint/512512.ckpt` to generate pictures(Or other models saved)
