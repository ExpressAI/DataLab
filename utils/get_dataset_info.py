from __future__ import annotations

import traceback
import argparse
import os
import sys
import importlib
import inspect

import json
from datalabs import GeneratorBasedBuilder, load_dataset

"""
This script writes out information regarding all datasets supported by the datalab
SDK in jsonl format, in general one dataset per line.
"""

# TODO(gneubig): there is probably an easier way of doing this without downloading
#                datasets, but it's non-trivial, so we'll stick with this for now
def get_splits(dataset: str, sub_dataset: str | None, prev_data: dict) -> list[str]:
    """
    Get the splits for each dataset and sub_dataset.
    :param dataset: the name of the dataset
    :param sub_dataset: the name of the sub-dataset, or none if none
    :param prev_data: previous data from the json file for caching
    :return: a list of split names
    """
    sub_str = sub_dataset if sub_dataset is not None else '__NONE__'
    if dataset in prev_data and prev_data[dataset] != 'ERROR' and sub_str in prev_data[dataset]['sub_datasets']:
        print(f'using cached splits for {dataset}, {sub_dataset}', file=sys.stderr)
        return prev_data[dataset]['sub_datasets'][sub_str]['splits']
    else:
        print(f'loading splits from datalab for {dataset}, {sub_dataset}', file=sys.stderr)
        loaded = load_dataset(dataset, sub_dataset)
        return list(loaded.keys())

def main():

    parser = argparse.ArgumentParser(description='Get dataset info and write it as json', allow_abbrev=False)
    parser.add_argument('--get_splits', help='get information about splits', action='store_true')
    parser.add_argument('--previous_jsonl', help='if a jsonl file already exists, read it in and cache information to save time', type=str, required=False, default=None)
    parser.add_argument('--output_jsonl', help='output jsonl file', required=False, type=str, default=None)
    args = parser.parse_args()

    prev_data = {}
    if args.previous_jsonl is not None:
        with open(args.previous_jsonl, 'r') as fin:
            for line in fin:
                for k, v in json.loads(line).items():
                    prev_data[k] = v

    dir_datasets = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../datasets/",
    )
    sys.path.append(dir_datasets)

    # This is for parent classes that are used across different datasets
    abstract_members = {'Wmt'}

    # Datasets to skip because they're too time-consuming to download the data.
    # We can download these manually?
    skip_datasets = {'wmt14', 'wmt15', 'wmt16', 'wmt17', 'wmt18', 'wmt19'}

    out_stream = sys.stdout if args.output_jsonl is None else open(args.output_jsonl, 'w')
    for file_name in sorted(os.listdir(dir_datasets)):
        print(f'---- {file_name} ----', file=sys.stderr)
        if not file_name.endswith(".py") and not file_name.endswith(".md") and not file_name.endswith(".pkl") and file_name!="__pycache__":
            try:
                my_module = importlib.import_module(f"{file_name}.{file_name}")

                metadata = {}
                for name, obj in inspect.getmembers(my_module):
                    if file_name in skip_datasets:
                        metadata = 'SKIPPED'
                        break
                    elif name in abstract_members:
                        continue

                    if inspect.isclass(obj) and issubclass(obj, GeneratorBasedBuilder):

                        dataset = obj()

                        dataset_info = dataset._info()
                        metadata["dataset_class_name"] = name
                        metadata["languages"] = dataset_info.languages
                        metadata["task_templates"] = None if dataset_info.task_templates is None else [x.task for x in dataset_info.task_templates]
                        sub_datasets = {}
                        if len(dataset.builder_configs) > 0:
                            for k, v in dataset.builder_configs.items():
                                sub_datasets[k] = {'splits': get_splits(file_name, k, prev_data)}
                        else:
                            sub_datasets['__NONE__'] = {'splits': get_splits(file_name, None, prev_data)}
                        metadata["sub_datasets"] = sub_datasets

                print(f'printing metadata for {file_name}', file=sys.stderr)
            except Exception as e:
                traceback.print_exc()
                metadata = 'ERROR'
            print(json.dumps({file_name: metadata}), file=out_stream)
            out_stream.flush()

if __name__ == '__main__':
    main()