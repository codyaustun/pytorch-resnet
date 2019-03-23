import os
import json
from collections import defaultdict

import pandas as pd

from resnet.cifar10.train import MODELS
from resnet.utils import count_parameters


MODEL_SIZES = {key: count_parameters(MODELS[key]()) for key in MODELS.keys()}


def single_run_acc(df, key='epoch'):
    df = df.copy()
    if 'prev_timestamp' in df:
        df['duration'] = (df['timestamp'] - df['prev_timestamp']).apply(lambda x: x.total_seconds())  # noqa: E501
    df['batch_duration'] = df['batch_duration'].apply(lambda x: x.total_seconds())  # noqa: E501

    _columns = [key, 'batch_size', 'ncorrect', 'batch_duration']
    if 'prev_timestamp' in df:
        _columns += ['duration']

    tmp = df.loc[:, _columns].groupby(key).sum()
    tmp['accuracy'] = tmp['ncorrect'] / tmp['batch_size']
    tmp['_throughput'] = tmp['batch_size'] / tmp['batch_duration']
    if 'prev_timestamp' in df:
        tmp['throughput'] = tmp['batch_size'] / tmp['duration']
    tmp['elapsed'] = df.groupby(key)['elapsed'].agg('max')
    tmp.reset_index(inplace=True)

    return tmp


def load_file(file, start_timestamp=None):
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['batch_duration'] = pd.to_timedelta(df['batch_duration'])
    df['ncorrect'] = df['top1_correct']
    start_timestamp = start_timestamp or df['timestamp'].iloc[0]
    df['elapsed'] = df['timestamp'] - start_timestamp
    df['batch_accuracy'] = df['ncorrect'] / df['batch_size']
    return df


def load_data(directory, verbose=True, key='epoch', prev_timestamp=True,
              splits=('train', 'valid', 'test')):

    # Put train first if prev_timestamp is true
    frames = []
    start_timestamp = None
    for index, split in enumerate(splits):
        try:
            path = os.path.join(directory, '{}_results.csv'.format(split))
            data = load_file(path, start_timestamp=start_timestamp)
            data['mode'] = split
            frames.append(data)
            if prev_timestamp and index == 0:
                start_timestamp = data['timestamp'].iloc[0]
            if verbose:
                print(path)
                print("{} results shape: {}".format(split, data.shape))
        except FileNotFoundError:
            if verbose:
                print("{} split doesn't exist: {}".format(split, path))

    if len(frames) > 0:
        combined = pd.concat(frames, ignore_index=True).sort_values(by='timestamp')  # noqa: E501
        combined['prev_timestamp'] = combined['timestamp'].shift(1)
        combined.loc[0, 'prev_timestamp'] = combined.loc[0, 'timestamp'] - combined.loc[0, 'batch_duration']  # noqa: E501

        result = {}
        for split in combined['mode'].unique():
            data = combined[combined['mode'] == split].copy()
            result[split] = single_run_acc(data, key=key)

        return result
    return {}


def load_multiple(directory, timestamps=None, verbose=False, key='epoch',
                  unpack=True,
                  prev_timestamp=True, splits=('train', 'valid', 'test')):

    timestamps = timestamps or os.listdir(directory)
    results = defaultdict(list)
    for timestamp in sorted(timestamps):
        _dir = os.path.join(directory, timestamp)
        result = load_data(_dir, verbose=verbose, key=key, splits=splits,
                           prev_timestamp=prev_timestamp)
        for split, data in result.items():
            data['run'] = _dir
            data['job_start'] = timestamp
            results[split].append(data)

    results = {split: pd.concat(frames) for split, frames in results.items()}
    if unpack:
        # For backwards compatibility
        return results['train'], results['test']
    else:
        return results


def load_multiple_models(directory, verbose=False, key='epoch',
                         unpack=True,
                         prev_timestamp=True,
                         splits=('train', 'valid', 'test')):
    paths = os.listdir(directory)

    results = defaultdict(list)
    for model in sorted(paths):
        if verbose:
            print(f"Loading {model}")
        _dir = os.path.join(directory, model)
        model_result = load_multiple(_dir, verbose=verbose, splits=splits,
                                     key=key, prev_timestamp=prev_timestamp,
                                     unpack=False)
        for split, data in model_result.items():
            data['model'] = model
            if model in MODELS:
                data['nparameters'] = MODEL_SIZES[model]
            results[split].append(data)

    results = {split: pd.concat(frames) for split, frames in results.items()}
    if unpack:
        # For backwards compatibility
        return results['train'], results['test']
    else:
        return results


def concat_update(existing, other, repeat=False):
    for key in other.keys():
        if key in existing:
            if existing[key] != other[key] or repeat:
                current = existing[key]
                if isinstance(current, list):
                    current.append(other[key])
                else:
                    existing[key] = [current, other[key]]
        else:
            existing[key] = other[key]


def run_config(run, repeat=False):
    full = {}
    configs = (os.path.join(run, entry.name) for entry in os.scandir(run) if 'config' in entry.name)

    for config in sorted(configs):
        with open(config) as file:
            tmp = json.load(file)

        tmp['path'] = config
        concat_update(full, tmp, repeat=repeat)
    return full


def search_configs(criteria, configs):
    matches = []
    for run, config in configs.items():
        is_match = True
        for key, value in criteria.items():
            try:
                config_value = config[key]
                if config_value != value:
                    is_match = False
            except KeyError:
                is_match = False

        if is_match:
            matches.append(run)

    return matches
