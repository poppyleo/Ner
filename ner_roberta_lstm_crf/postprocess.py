# /usr/bin/env python
# coding=utf-8
"""输出submit.json"""
import json

from metrics import get_entities
import utils

params = utils.Params()

IO2str = {
    'EXP': '试验要素',
    'PER': '性能指标',
    'SYS': '系统组成',
    'SCE': '任务场景'
}


def get_submit():
    # read tags.txt to dict
    with open(params.data_dir / 'test/tags_pre.txt', 'r') as f:
        submit = {}
        for idx, line in enumerate(f):
            entities = get_entities(line.strip().split(' '))
            sample_list = []
            for entity in entities:
                enti_dict = {'label_type': None, 'overlap': 0, 'start_pos': None, 'end_pos': None}
                enti_dict['label_type'] = IO2str[entity[0].strip()]
                enti_dict['start_pos'] = entity[1] + 1
                enti_dict['end_pos'] = entity[2] + 1
                sample_list.append(enti_dict)
            submit[f"validate_V2_{idx + 1}.json"] = sample_list

        # convert dict to json
        with open(params.data_dir / 'submit.json', 'w', encoding='utf-8') as w:
            json_data = json.dumps(submit, indent=4, ensure_ascii=False)
            w.write(json_data)


if __name__ == '__main__':
    get_submit()
