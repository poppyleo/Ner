# /usr/bin/env python
# coding=utf-8
"""Predict"""

import argparse
import random
import logging
import os

import torch

from transformers import ElectraConfig, RobertaConfig

from model import ElectraForTokenClassification
import utils
from dataloader import NERDataLoader

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--restore_file', default=None, required=False,
                    help="Optional, name of the file containing weights to reload before training")
parser.add_argument('--mode', default='test', help="'val' or 'test'")


def predict(model, data_iterator, params, mode):
    """Predict entities
    """
    # set model to evaluation mode
    model.eval()

    # id2tag dict
    idx2tag = {idx: tag for idx, tag in enumerate(params.tags)}

    pred_tags = []

    for input_ids, input_mask, labels in data_iterator:
        # to device
        input_ids = input_ids.to(params.device)
        input_mask = input_mask.to(params.device)
        # inference
        with torch.no_grad():
            batch_output = model(input_ids, attention_mask=input_mask)
        # List[List[str]]
        pred_tags.extend([[idx2tag.get(idx) for idx in indices] for indices in batch_output])

    # write to file
    with open(params.data_dir / mode / 'tags_pre.txt', 'w', encoding='utf-8') as file_tags:
        for tag in pred_tags:
            file_tags.write('{}\n'.format(' '.join(tag)))


if __name__ == '__main__':
    args = parser.parse_args()
    # 设置模型使用的gpu
    torch.cuda.set_device(7)
    # 查看现在使用的设备
    print('current device:', torch.cuda.current_device())
    # 预测验证集还是测试集
    mode = args.mode
    params = utils.Params()
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    utils.set_logger()

    # Create the input data pipeline
    logging.info("Loading the dataset...")
    dataloader = NERDataLoader(params)
    val_loader, test_loader = dataloader.load_data(mode='test')
    logging.info("- done.")

    # Define the model
    logging.info('Loading the model...')
    config_path = os.path.join(params.params_path, 'bert_config.json')
    config = RobertaConfig.from_json_file(config_path)
    model = ElectraForTokenClassification(config, params=params)

    model.to(params.device)
    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'), model)
    logging.info('- done.')

    logging.info("Starting prediction...")
    if mode == 'test':
        predict(model, test_loader, params, mode=mode)
    elif mode == 'val':
        predict(model, val_loader, params, mode=mode)
    logging.info('- done.')

    # 检查文本和预测结果长度是否一致
    with open(params.data_dir / mode / 'tags_pre.txt', 'r') as f_t, \
            open(params.data_dir / mode / 'sentences.txt', 'r') as f_s:
        for t, s in zip(f_t, f_s):
            assert len(s.split(' ')) == len(t.split(' ')), "Length of test text is not equal to predicted tags!"
