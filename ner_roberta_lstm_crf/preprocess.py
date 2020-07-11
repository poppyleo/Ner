# -!- coding: utf-8 -!-
"""get train and test dataset"""
import json
import copy
from pathlib import Path
from utils import Params

params = Params()


def alldata2list(train_data_path):
    """将所有训练数据合并到一个列表
    :return: train_data (List[Dict])
    """
    train_data = []
    for idx in range(1, 401):
        with open(train_data_path / f'train_V2_{idx}.json', encoding='gbk') as f:
            data = json.load(f)
            train_data.append(data)
    return train_data


def construction(save_dir, train_data):
    """构造sentence.txt和与其对应的tags.txt
    @:param train_data (List[Dict])
    """
    # 保存所有的句子和标签
    data_all_text = []
    data_all_tag = []

    for data in train_data:
        # 取文本和标注
        # 去掉原文本前后的回车换行符
        # 将原文本中间的回车换行符替换成r（符合源数据标注规则）
        # 将特殊字符替换为UNK
        data_ori = list(data['originalText'].strip().replace('\r\n', '✄').replace(' ', '✄'))
        data_text = copy.deepcopy(data_ori)
        data_entities = data['entities']

        for entity in data_entities:
            # 取当前实体类别
            en_type = entity['label_type']
            # 取当前实体标注
            en_tags = params.ne_dict[en_type]  # ['B-XXX', 'I-XXX']
            start_ind = entity['start_pos'] - 1
            end_ind = entity['end_pos'] - 1
            # 替换实体
            data_text[start_ind] = en_tags[0]
            data_text[start_ind + 1:end_ind + 1] = [en_tags[1] for _ in range(end_ind - start_ind)]
        # 替换非实体
        for idx, item in enumerate(data_text):
            # 如果元素不是已标注的命名实体
            if len(item) != 5:
                data_text[idx] = params.ne_dict['Others']
        assert len(data_ori) == len(data_text), f'生成的标签与原文本长度不一致！'
        data_all_text.append(data_ori)
        data_all_tag.append(data_text)

    assert len(data_all_text) == len(data_all_tag), '样本数不一致！'
    # 写入训练集
    with open(save_dir / 'train/sentences.txt', 'w', encoding='utf-8') as file_sentences, \
            open(save_dir / 'train/tags.txt', 'w', encoding='utf-8') as file_tags:
        # 逐行对应写入
        for sentence, tag in zip(data_all_text[:350], data_all_tag[:350]):
            file_sentences.write('{}\n'.format(' '.join(sentence)))
            file_tags.write('{}\n'.format(' '.join(tag)))

    # 写入验证集
    with open(save_dir / 'val/sentences.txt', 'w', encoding='utf-8') as file_sentences, \
            open(save_dir / 'val/tags.txt', 'w', encoding='utf-8') as file_tags:
        # 逐行对应写入
        for sentence, tag in zip(data_all_text[350:], data_all_tag[350:]):
            file_sentences.write('{}\n'.format(' '.join(sentence)))
            file_tags.write('{}\n'.format(' '.join(tag)))


def get_testset(data_path, save_dir):
    """获取测试集
    """
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)
    # 将特殊字符替换为UNK
    data = [(key, sen.strip().replace('\r\n', '✄').replace(' ', '✄').replace('\x1a', '✄')) for key, sen in data.items()]
    # 根据序号排序
    data = sorted(data, key=lambda d: int(d[0].split('_')[-1].split('.')[0]))
    # 写入训练集
    with open(save_dir / 'test/sentences.txt', 'w', encoding='utf-8') as file_sentences:
        # 逐行对应写入
        for _, sentence in data:
            file_sentences.write('{}\n'.format(' '.join(sentence)))


def generate_test_tags(params):
    """生成测试集标签
    """
    # generate text length
    with open(params.data_dir / 'test/sentences.txt', 'r', encoding='utf-8') as f:
        text_len = [len(line.strip().split(' ')) for line in f]
    # generate tags
    with open(params.data_dir / 'test/tags.txt', 'w', encoding='utf-8') as f:
        for l in text_len:
            tag = ['O' for _ in range(l)]
            f.write('{}\n'.format(' '.join(tag)))
    # sanity check
    with open(params.data_dir / 'test/sentences.txt', 'r', encoding='utf-8') as file_sens, \
            open(params.data_dir / 'test/tags.txt', 'r', encoding='utf-8') as file_tags:
        for sen, tag in zip(file_sens, file_tags):
            assert len(sen.strip().split(' ')) == len(tag.strip().split(' ')), 'Mismatched! check test set!'


if __name__ == '__main__':
    train_data_path = Path('./ccks_8_data_v2/train')
    train_data = alldata2list(train_data_path)
    save_dir = Path('./data')
    construction(save_dir, train_data)
    test_data_path = Path('./ccks_8_data_v2/validate_data.json')
    get_testset(test_data_path, save_dir)
    generate_test_tags(params)
