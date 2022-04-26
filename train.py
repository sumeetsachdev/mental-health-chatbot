import os
import math
from pprint import pformat
from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler,OptimizerParamsHandler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                                GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)


##from utils import get_dataset

SPECIAL_TOKENS = ['<bos>', '<eos>', '<speaker1>', '<speaker2>', '<pad>']
ATTR_TO_SPECIAL_TOKENS = {
    'bos_token': '<bos>',
    'eos_token': '<eos>',
    'pad_token': '<pad>',
     'additional_special_tokens': ['<speaker1>', '<speaker2>']
    }
MODEL_INPUTS = ['input_ids', 'mc_token_ids', 'lm_labels', 'mc_labels', 'token_type_ids']
PADDED_INPUTS = ['input_ids', 'lm_labels', 'token_type_ids']


LOCAL_RANK = -1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)



def average_distributed_scalar(scalar, local_rank):
    if local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def pad_dataset(dataset, padding=0):
    max_l = max(len(x) for x in dataset['input_ids'])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != 'lm_labels' else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def add_special_tokens(model, tokenizer):
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [ [bos] + list(chain(*persona)) ] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [ [speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i,s in enumerate(sequence[1:]) ]
    instance = {}
    instance['input_ids'] = list(chain(*sequence))
    instance['token_type_ids'] = [ speaker2 if i % 2 else speaker1 for i,s in enumerate(sequence) for _ in s ]
    instance['mc_token_ids'] = len(instance['input_ids']) - 1
    instance['lm_labels'] = [-100] * len(instance['input_ids'])
    if lm_labels:
        instance['lm_labels'] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance
    

def get_data_loaders(dataset_path, dataset_cache, num_candidates, personality_permutations, max_history, distributed, tokenizer):
    personachat: get_dataset(tokenizer, dataset_path, dataset_cache)
    datasets = ['train': defaultdict(list), 'valid':defaultdict(list)]
    for dataset_name, dataset in persona_chat.items():
        min_num_candidates = len(dataset[0]['utterances'][0]['candidates'])
        if num_candidates > 0 and dataset_name == 'train':
            min_num_candidates = min(num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog['personality'].copy()
            for _ in range(personality_permutations):
                for utterance in dialog['utterances']:
                    history = utterance['history'][-(2*max_history+1):]
                    for j, candidate in enumerate(utterance['candidates'][-min_num_candidates:]):
                        lm_labels = bool(j == min_num_candidates-1)
                        instance = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]['mc_labels'].append(min_num_candidates - 1)
                    datasets[dataset_name]['n_candidates'] = min_num_candidates
                persona = [persona[-1]] + persona[:-1]
    tensor_datasets = {'train': [],
                       'valid': []}
    for dataset_name, dataset in dataset.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != 'mc_labels':
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:]))
            tensor_datasets[dataset_name].append(tensor)


    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)
    return train_loader, valid_loader, train_sampler, valid_sampler
    
