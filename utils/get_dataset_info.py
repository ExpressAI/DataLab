from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import traceback

from datalabs import GeneratorBasedBuilder, load_dataset

"""
This script writes out information regarding all datasets supported by the datalab
SDK in jsonl format, in general one dataset per line.
"""

# TODO(gneubig): there is probably an easier way of doing this without downloading
# datasets, but it's non-trivial, so we'll stick with this for now


def get_splits(dataset: str, sub_dataset: str | None) -> dict[str, int]:
    """
    Get the splits for each dataset and sub_dataset.
    :param dataset: the name of the dataset
    :param sub_dataset: the name of the sub-dataset, or none if none
    :param prev_data: previous data from the json file for caching
    :return: a list of split names
    """
    # sub_str = sub_dataset if sub_dataset is not None else "__NONE__"
    print(f"loading splits from datalab for {dataset}, {sub_dataset}", file=sys.stderr)
    loaded = load_dataset(dataset, sub_dataset)
    return {k: v.num_rows for k, v in loaded.items()}


def main():

    parser = argparse.ArgumentParser(
        description="Get dataset info and write it as json", allow_abbrev=False
    )
    parser.add_argument(
        "--get_splits", help="get information about splits", action="store_true"
    )
    parser.add_argument(
        "--previous_jsonl",
        help="if a jsonl file already exists, read it in and cache "
        "information to save time",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--output_jsonl",
        help="output jsonl file",
        required=False,
        type=str,
        default=None,
    )
    args = parser.parse_args()

    prev_data = {}
    if args.previous_jsonl is not None:
        with open(args.previous_jsonl, "r") as fin:
            for line in fin:
                for k, v in json.loads(line).items():
                    prev_data[k] = v

    dir_datasets = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../datasets/",
    )
    sys.path.append(dir_datasets)

    # This is for parent classes that are used across different datasets
    abstract_members = {"Wmt"}

    # Datasets to skip because they're too time-consuming to download the data.
    # trivia_qa is skipped due to https://github.com/ExpressAI/DataLab/issues/200
    # weibo4_moods is skipped because downloading stalled
    # wmt were too large and time consuming (for now)
    skip_datasets = {
        "trivia_qa",
        "weibo4_moods",
        "wmt14",
        "wmt15",
        "wmt16",
        "wmt17",
        "wmt18",
        "wmt19",
    }

    out_stream = (
        sys.stdout if args.output_jsonl is None else open(args.output_jsonl, "w")
    )
    for file_name in sorted(os.listdir(dir_datasets)):
        print(f"---- {file_name} ----", file=sys.stderr)
        if (
            not file_name.endswith(".py")
            and not file_name.endswith(".md")
            and not file_name.endswith(".pkl")
            and file_name != "__pycache__"
        ):

            try:

                my_module = importlib.import_module(f"{file_name}.{file_name}")

                metadata = {}
                for name, obj in inspect.getmembers(my_module):
                    if file_name in skip_datasets:
                        print(
                            json.dumps({f"{file_name}---__NONE__": "SKIPPED"}),
                            file=out_stream,
                        )
                        break
                    elif name in abstract_members:
                        continue

                    if inspect.isclass(obj) and issubclass(obj, GeneratorBasedBuilder):

                        config_names = (
                            [None]
                            if len(obj.builder_configs) == 0
                            else [x for x in obj.builder_configs]
                        )

                        for sub_dataset in config_names:

                            dataset_id = f"{file_name}---{sub_dataset}"

                            # Use cached data if it exists
                            if dataset_id in prev_data and isinstance(
                                prev_data[dataset_id], dict
                            ):
                                print(
                                    f"printing cached metadata "
                                    f"for {file_name} {sub_dataset}",
                                    file=sys.stderr,
                                )
                                print(
                                    json.dumps({dataset_id: prev_data[dataset_id]}),
                                    file=out_stream,
                                )
                                continue

                            dataset = obj(name=sub_dataset)

                            dataset_info = dataset._info()
                            metadata["dataset_name"] = file_name
                            metadata["dataset_class_name"] = name
                            if sub_dataset:
                                metadata["sub_dataset"] = sub_dataset
                            metadata["splits"] = get_splits(file_name, sub_dataset)

                            metadata["languages"] = dataset_info.languages
                            if dataset_info.task_templates is not None:
                                metadata["task_categories"] = [
                                    x.task_category for x in dataset_info.task_templates
                                ]
                                metadata["tasks"] = [
                                    x.task for x in dataset_info.task_templates
                                ]

                            print(
                                f"printing metadata for {file_name} {sub_dataset}",
                                file=sys.stderr,
                            )
                            print(json.dumps({dataset_id: metadata}), file=out_stream)
                            out_stream.flush()
            except Exception as e:  # noqa
                traceback.print_exc()
                print(json.dumps({f"{file_name}---__NONE__": "ERROR"}), file=out_stream)


if __name__ == "__main__":
    main()
