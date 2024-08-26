import random
from tqdm import tqdm 
from typing import List, Dict, Tuple
from datasets import load_dataset

def select_data(data_name, train=True):
    train_data, val_data, test_data = load_data(data_name)
    if not train:
        return test_data
    return val_data, train_data

def load_data(data_name:str, cache_dir:str=None, lim:int=None)->Tuple['train', 'val', 'test']:
    data_ret = {
        'squad'   : _squad,
    }
    return data_ret[data_name](cache_dir, lim)

def _squad(cache_dir, lim:int=None)->List[Dict['context', 'question']]:
    dataset = load_dataset("rajpurkar/squad", cache_dir=cache_dir)
    train = list(dataset['train'])[:lim]
    train = [{'context':t['context'], 'question':t['question']} for t in train]
    test = list(dataset['validation'])[:lim]
    test = [{'context':t['context'], 'question':t['question']} for t in test]
    return train, test, test