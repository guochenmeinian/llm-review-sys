Here we learned to train the models using LLama-Factory and writing our own code implementation to train to get ourseleves familar with how the pipeline works and the related technologies.


In case AutoDL has errors, use the following codes:
```bash
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# if issue persists
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
```


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
