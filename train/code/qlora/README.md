
```
accelerate launch --multi_gpu qlora_train_accelerate.py --config_file qlora_train_config.yaml
```

这里用到了deepspeed来加速 用了5张4090 跑这个代码：
```
deepspeed --num_gpus=5 qlora_train.py --config_file qlora_train_config.yaml
```
AutoDL不支持的话要来这个HF镜像：https://hf-mirror.com/
```
HF_ENDPOINT=https://hf-mirror.com deepspeed qlora_ds_train.py --config_file qlora_train_config.yaml --deepspeed qlora_ds_config.json
```

