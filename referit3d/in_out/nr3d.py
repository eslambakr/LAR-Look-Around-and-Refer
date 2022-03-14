import json
import pandas as pd
import os.path as osp
import pathlib

from ..data_generation.nr3d import decode_stimulus_string


def dummy_lambda(x):
    return x.replace('_', ' ')


def dummy_lambda_1(x):
    return int(x)


def dummy_lambda_2(x):
    return decode_stimulus_string(x)[0]

def dummy_lambda_3(x):
    return decode_stimulus_string(x)[1]

def dummy_lambda_4(x):
    return decode_stimulus_string(x)[3]

def dummy_lambda_5(x):
    return x in ['clothes', 'clothing']
def load_scan_refer_data_like_refer_it_3d(scan_ref_data_dir, merge_train_val=True):
    sr_data = dict()
    for split in ['train', 'val']:
        sr_file = osp.join(scan_ref_data_dir, 'ScanRefer_filtered_{}.json'.format(split))
        with open(sr_file) as fin:
            sr_data_temp = json.load(fin)
        sr_data_temp = pd.DataFrame.from_dict(sr_data_temp)
        #sr_data_temp.object_name = sr_data_temp.object_name.apply(lambda x: x.replace('_', ' '))
        sr_data_temp.object_name = sr_data_temp.object_name.apply(dummy_lambda)
        sr_data_temp.rename(columns={'object_name': 'instance_type',
                                     'description': 'utterance',
                                     'object_id': 'target_id'},
                            inplace=True)
        sr_data_temp.drop(columns=['ann_id', 'token'], inplace=True)
        sr_data_temp['target_id'] = sr_data_temp.target_id.apply(dummy_lambda_1)
        #sr_data_temp['target_id'] = sr_data_temp.target_id.apply(lambda x: int(x))
        sr_data[split] = sr_data_temp

    if merge_train_val:
        sr_data = pd.concat(sr_data.values(), axis=0)

    return sr_data

def load_nr3d_raw_data(refer_it_csv, drop_bad_context=True):
    df = pd.read_csv(refer_it_csv)
    df.rename(columns={'Input.stimulus_id': 'stimulus_id', 'Answer.response': 'utterance'}, inplace=True)
    df['scan_id'] = df.stimulus_id.apply(dummy_lambda_2)
    df['instance_type'] = df.stimulus_id.apply(dummy_lambda_3)
    df['target_id'] = df.stimulus_id.apply(dummy_lambda_4)
    # df['scan_id'] = df.stimulus_id.apply(lambda x: decode_stimulus_string(x)[0])
    # df['instance_type'] = df.stimulus_id.apply(lambda x: decode_stimulus_string(x)[1])
    # df['target_id'] = df.stimulus_id.apply(lambda x: decode_stimulus_string(x)[3])

    if drop_bad_context:
        basedir = osp.split(pathlib.Path(__file__).parent.absolute())[0]
        bad_context = osp.join(basedir, 'data/language/nr3d/manually_inspected_bad_contexts.csv')
        bad_context = pd.read_csv(bad_context)
        bad_context = set(bad_context['stimulus_id'].unique())
        drop_mask = df.instance_type.apply(dummy_lambda_5)

        #drop_mask = df.instance_type.apply(lambda x: x in ['clothes', 'clothing'])
        drop_mask |= df.stimulus_id.isin(bad_context)
        print('dropping ', (drop_mask.sum()), 'utterances marked manually as bad/poor context')
        df = df[~drop_mask]
        df.reset_index(inplace=True, drop=True)

    return df
