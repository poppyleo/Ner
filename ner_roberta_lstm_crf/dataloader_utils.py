#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataloader.py utils"""


class InputExample(object):
    """a single set of samples of data_src
    """

    def __init__(self, sentence, tags):
        """
        Desc:
            is_impossible: bool, [True, False]
        """
        self.sentence = sentence
        self.tags = tags


class InputFeatures(object):
    """
    Desc:
        a single set of features of data_src
    Args:
        start_pos: start position is a list of symbol
        end_pos: end position is a list of symbol
    """

    def __init__(self,
                 tokens,
                 input_ids,
                 input_mask,
                 tag
                 ):
        self.tokens = tokens
        self.input_mask = input_mask
        self.input_ids = input_ids
        self.tag = tag


def read_examples(data_dir, data_sign):
    """read BIO-NER data_src to InputExamples
    :return examples (List[InputExample]):
    """
    examples = []
    # read src data
    with open(data_dir / f'{data_sign}/sentences.txt', "r", encoding='utf-8') as f_sen, \
            open(data_dir / f'{data_sign}/tags.txt', 'r', encoding='utf-8') as f_tag:
        for sen, tag in zip(f_sen, f_tag):
            example = InputExample(sentence=sen.strip().split(' '), tags=tag.strip().split(' '))
            examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def convert_examples_to_features(params, examples, tokenizer, pad_sign=True):
    """convert examples to features.
    :param examples (List[InputExamples]): data_src examples.
    :param pad_sign: 是否补零
    """
    # tag to id
    tag2idx = {tag: idx for idx, tag in enumerate(params.tags)}
    features = []

    # context max len
    max_len = params.max_seq_length

    for (example_idx, example) in enumerate(examples):
        # tokenize test
        text_tokens = [tokenizer.tokenize(w)[0] for w in example.sentence]
        # label id
        tag_idx = [tag2idx[tag] for tag in example.tags]

        # cut off
        if len(text_tokens) >= max_len:
            text_tokens = text_tokens[:max_len]
            tag_idx = tag_idx[:max_len]

        # token to id
        text_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        # zero-padding up to the sequence length
        if len(text_tokens) < max_len and pad_sign:
            # 补零
            pad_len = max_len - len(text_ids)
            text_ids += [tokenizer.pad_token_id] * pad_len
            tag_idx += [tag2idx['O']] * pad_len
        # mask
        input_mask = [1 if idx > 0 else 0 for idx in text_ids]

        # get features
        features.append(
            InputFeatures(
                tokens=text_tokens,
                input_ids=text_ids,
                input_mask=input_mask,
                tag=tag_idx
            ))

    return features
