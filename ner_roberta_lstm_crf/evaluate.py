# /usr/bin/env python
# coding=utf-8
"""Evaluate the model"""
import argparse
import logging

import torch

import utils
from metrics import f1_score, accuracy_score
from metrics import classification_report
from metrics import get_entities
from postprocess import IO2str

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--type', default='val', help="'val' or 'test'")


def evaluate(model, data_iterator, params, mark='Eval', verbose=True):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    # id2tag dict
    idx2tag = {idx: tag for idx, tag in enumerate(params.tags)}

    true_tags = []
    pred_tags = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()
    for input_ids, input_mask, labels in data_iterator:
        # to device
        input_ids = input_ids.to(params.device)
        input_mask = input_mask.to(params.device)
        labels = labels.to(params.device)

        batch_size, max_len = labels.size()

        # get loss
        loss = model(input_ids, attention_mask=input_mask.bool(), labels=labels)
        loss /= batch_size
        # update the average loss
        loss_avg.update(loss.item())

        # inference
        with torch.no_grad():
            batch_output = model(input_ids, attention_mask=input_mask.bool())

        # 恢复标签真实长度
        real_batch_tags = []
        for i in range(batch_size):
            real_len = int(input_mask[i].sum())
            real_batch_tags.append(labels[i][:real_len].to('cpu').numpy())

        # List[int]
        pred_tags.extend([idx2tag.get(idx) for indices in batch_output for idx in indices])
        true_tags.extend([idx2tag.get(idx) for indices in real_batch_tags for idx in indices])
    # sanity check
    assert len(pred_tags) == len(true_tags), 'len(pred_tags) is not equal to len(true_tags)!'

    # logging loss, f1 and report
    metrics = {}
    f1 = f1_score(true_tags, pred_tags)
    accuracy = accuracy_score(true_tags, pred_tags)
    metrics['loss'] = loss_avg()
    metrics['f1'] = f1
    metrics['accuracy'] = accuracy
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    # f1 classification report
    if verbose:
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics


def get_type_entity(f, sentences):
    """获取实体类别和文本
    :param f: 标签文件
    :param sentences (List[List[str]]): 文本
    :return: result: 实体类别和文本
    """
    result = []
    for idx, line in enumerate(f):
        # get BIO-tag
        entities = get_entities(line.strip().split(' '))
        sample_list = []
        for entity in entities:
            label_type = IO2str[entity[0]]
            start_ind = entity[1]
            end_ind = entity[2]
            en = sentences[idx][start_ind:end_ind + 1]
            sample_list.append({label_type: ''.join(en)})
        result.append(sample_list)
    return result


def analyze_result(params, type):
    """分析文本形式结果
    :param type: 'test' or 'val'
    """
    # get text
    with open(params.data_dir / type / 'sentences.txt', 'r', encoding='utf-8') as f:
        sentences = [line.strip().split(' ') for line in f]
    # 真实标签
    if type == 'val':
        with open(params.data_dir / 'val/tags.txt', 'r') as f:
            re_true = get_type_entity(f, sentences)
    # 预测标签
    with open(params.data_dir / f'{type}/tags_pre.txt', 'r') as f:
        re_pre = get_type_entity(f, sentences)
    if type == 'val':
        return re_true, re_pre
    else:
        return re_pre


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params()
    if args.type == 'val':
        re_true, re_pre = analyze_result(params, type=args.type)
    else:
        re_pre = analyze_result(params, type=args.type)

    if args.type == 'val':
        for idx, (t, p) in enumerate(zip(re_true, re_pre)):
            print(f'id={idx + 1}:')
            print('真实标签：', t)
            print('预测标签：', p)
            print('-' * 50)
    else:
        for idx, p in enumerate(re_pre):
            print(f'id={idx + 1}:')
            print('预测标签：', p)
            print('-' * 50)
