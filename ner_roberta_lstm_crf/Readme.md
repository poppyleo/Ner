# NER FOR EXPERIMENT
## Usage:

预处理：`python preprocess.py`

训练：`python train.py --epoch_num= [--restore_file=]` 

预测：`python predict.py --restore_file= --type='test' or 'val'`

获取提交文件：`python postprocess.py`

分析预测结果：`python evaluate.py --type='val' or 'test'`

## 实体对应：

|  Tag  | Meaning                                          |
| :---: | ------------------------------------------------ |
|   O   | 非实体                    |
| B-EXP | 实验要素开头               |
| I-EXP | 实验要素非开头             |
| B-PER | 性能指标开头          |
| I-PER | 性能指标非开头     |
| B-SYS | 系统组成开头           |
| I-SYS | 系统组成非开头       |
| B-SCE | 任务场景开头     |
| I-SCE | 任务场景非开头 |

## 后处理

TODO：单字实体与上下文多字实体拼接