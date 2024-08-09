# 🦜 Multi-ARCL: Multimodal Adaptive Relay-based Distributed Continual Learning for Encrypted Traffic Classification 

<p align="center">
  <a href="#-introduction">🎉Introduction</a> •
  <a href="#-model">🦜Model</a> •
  <a href="#-train">🔥Train</a> •
  <a href="#-datasets">🌟Datasets</a> •
  <a href="#-quick-start">📍Quick Start</a> •
  <a href="#-acknowledgement">👨‍🏫Acknowledgement</a> •  
  <a href="#-contact">🤗Contact</a>
</p>


## 🎉 Introduction
CL enables models to learn new knowledge without forgetting old applications, thus making TC models adaptable to network environment changes. For this purpose, scholars have re- searched the implementation of CL to maintain the precision of previous classifications while reducing training expenses. It is important to note that the network environment is ex- ceedingly complex, marked by the frequent emergence and removal of applications. Google typically removes apps from its market quarterly, reducing the number of available Android apps. In this context, we categorize the removed applications as silent applications and those that remain as active applications. Figure 1 shows the number of silent applications over the past year from AppBrain, with about one million apps declining in status daily. Moreover, in our study, we train a classifier to identify existing apps and test them in a new environment, focusing on category changes. Figure 1 also clearly demonstrates that a significant number of active applications are mistakenly identified as silent ones, which substantially impairs the model’s effectiveness. CL currently focuses on learning new applications and retaining old knowledge, but overlooks the capability of machine unlearning for silent applications. 

## 🦜 Model
<div align="center">
  <img src="./images/workflow.png" width="800px" />
</div>

## 🌟Datasets
[njupt2023](https://github.com/NJUPTSecurityAI/total-papers-summary/blob/main/njupt2023.csv),
[MIRAGE-2019](https://traffic.comics.unina.it/mirage/mirage-2019.html),
[CIC-IDS](https://www.unb.ca/cic/datasets/vpn.html)

## 🔥 Train

if you use parallel computing, please use this code：
two gpus
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 secondmain.py
```
four gpus
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 secondmain.py
```
if you find the error: 
```
File "/home/vipuser/miniconda3/lib/python3.8/site-packages/torch/distributed/rendezvous.py", line 179, in _env_rendezvous_handler
    store = TCPStore(master_addr, master_port, world_size, start_daemon, timeout)
RuntimeError: Address already in use
...
```
use this command:
```
kill $(lsof -t -i:12359)
```
and then reuse parallel computing.


## 👨‍🏫 Acknowledgement
This document is the results of the research project funded by National Natural Science Fundation (General Program) Grant No.61972211, China, University Industry Academy Research Innovation Fund No.2021FNA02006, China, Open trial of CENI-based network attack and defense exercise service platform No.2023C0302 and Development of an Ultra-large-scale Ubiquitous Network Quality Monitoring System Based on Trusted Edge Intelligence under Grant SYG202311.


## 🤗 Contact

If there are any questions, please feel free to propose new features by opening an issue or contacting the author: 2022040506@njupt.edu.cn
