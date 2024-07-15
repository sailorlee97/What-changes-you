if you use parallel computing, please use this code：

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 secondmain.py
```
