# Temporally-language-grounding
A Pytorch implemention for some state-of-the-art models for "Temporally language grounding in untrimmed videos"

## Three Models for this task
### Supervised Learning based methods
- TALL: Temporal Activity Localization via Language Query [pdf](http://openaccess.thecvf.com/content_ICCV_2017/papers/Gao_TALL_Temporal_Activity_ICCV_2017_paper.pdf)
- MAC: MAC: Mining Activity Concepts for Language-based Temporal Localization [pdf](https://arxiv.org/pdf/1811.08925.pdf)
### Reinforcement Learning based method
- A2C: Read, Watch, and Move: Reinforcement Learning for Temporally Grounding Natural Language Descriptions in Videos [pdf](https://arxiv.org/abs/1901.06829v1)

| Methods        | R@1, IoU0.7   |  R@1, IoU0.5  |
| --------   | -----:   | :----: |
| TALL        | $1      |   5    |
|  MAC        | $1      |   6    |
|  A2C        | $1      |   7    |


## Features Download

### Training and Testing for TALL
python main_charades_SL.py --model TALL

### Training and Testing for MAC
python main_charades_SL.py --model MAC

### Training and Testing for A2C
python main_charades_RL.py
