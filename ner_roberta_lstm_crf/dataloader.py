# /usr/bin/env python
# coding=utf-8
"""crf_cws dataloader"""

import os

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from dataloader_utils import read_examples, convert_examples_to_features


class NERDataLoader(object):
    """crf_cws dataloader
    """

    def __init__(self, params):
        self.params = params

        self.train_batch_size = params.train_batch_size
        self.dev_batch_size = params.dev_batch_size
        self.test_batch_size = params.test_batch_size

        self.data_dir = params.data_dir
        self.max_seq_length = params.max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=str(params.bert_model_dir),
                                                       do_lower_case=True,
                                                       never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
        # 保存数据(Bool)
        self.data_cache = params.data_cache

    def convert_examples_to_features(self, data_sign):
        """convert InputExamples to InputFeatures
        :param data_sign: 'train', 'val' or 'test'
        :return: features (List[InputFeatures]):
        """
        print("=*=" * 10)
        print("Loading {} data...".format(data_sign))

        # get examples
        if data_sign == "train":
            examples = read_examples(self.data_dir, data_sign='train')
        elif data_sign == "val":
            examples = read_examples(self.data_dir, data_sign='val')
        elif data_sign == "test":
            examples = read_examples(self.data_dir, data_sign='test')
        else:
            raise ValueError("please notice that the data can only be train/val/test !!")

        # get features
        # 数据保存路径
        cache_path = os.path.join(self.data_dir, "{}.cache.{}".format(data_sign, str(self.max_seq_length)))
        # 读取数据
        if os.path.exists(cache_path) and self.data_cache:
            features = torch.load(cache_path)
        else:
            # 生成数据
            features = convert_examples_to_features(self.params, examples, self.tokenizer)
            # save data
            if self.data_cache:
                torch.save(features, cache_path)
        return features

    def get_dataloader(self, data_sign="train"):
        """construct dataloader
        :param data_sign: 'train', 'val' or 'test'
        :return:
        """
        # InputExamples to InputFeatures
        features = self.convert_examples_to_features(data_sign=data_sign)

        print(f"{len(features)} {data_sign} data loaded!")
        print("=*=" * 10)
        # convert to tensor
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        labels = torch.tensor([f.tag for f in features], dtype=torch.long)
        dataset = TensorDataset(input_ids, input_mask, labels)

        # construct dataloader
        # RandomSampler(dataset) or SequentialSampler(dataset)
        if data_sign == "train":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size)
        elif data_sign == "val":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.dev_batch_size)
        elif data_sign == "test":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size)

        return dataloader

    def load_data(self, mode='train'):
        """load dataloader
        :param mode: 'train' or 'test'
        """
        if mode == 'train':
            train_dataloader = self.get_dataloader(data_sign="train")
            dev_dataloader = self.get_dataloader(data_sign="val")
            return train_dataloader, dev_dataloader
        elif mode == 'test':
            dev_dataloader = self.get_dataloader(data_sign="val")
            test_dataloader = self.get_dataloader(data_sign="test")
            return dev_dataloader, test_dataloader
        else:
            raise ValueError("please notice that the type can only be train/val/test !!")


if __name__ == '__main__':
    from utils import Params

    params = Params()
    datalodaer = NERDataLoader(params)
    f = datalodaer.convert_examples_to_features(data_sign='val')
    print(f[0].input_ids)
    print(f[0].input_mask)
    print(f[0].tag)
    print(f[0].tokens)
