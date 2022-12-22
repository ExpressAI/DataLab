from __future__ import annotations

from typing import cast

from datalabs import Dataset
from datalabs.features.features import ClassLabel, Sequence


def _replace_one(names: list[str], lab: int):
    return names[lab] if lab != -1 else "_NULL_"


def _replace_labels(features: dict, example: dict) -> dict:
    new_example = {}
    for examp_k, examp_v in example.items():
        examp_f = features[examp_k]
        # Label feature
        if isinstance(examp_f, ClassLabel):
            names = cast(ClassLabel, examp_f).names
            new_example[examp_k] = _replace_one(names, examp_v)
        # Sequence feature
        elif isinstance(examp_f, Sequence):
            examp_seq = cast(Sequence, examp_f)
            # Sequence of labels
            if isinstance(examp_seq.feature, ClassLabel):
                names = cast(ClassLabel, examp_seq.feature).names
                new_example[examp_k] = [_replace_one(names, x) for x in examp_v]
            # Sequence of anything else
            else:
                new_example[examp_k] = examp_v
        # Anything else
        else:
            new_example[examp_k] = examp_v
    return new_example


def recover_labels(dataset: Dataset):
    """
    Recover labels from the dataset.

    Args:
        dataset (Dataset): Dataset to recover labels from.
    Input:
    {'text': 'if you sometimes like to go to the movies to have fun ,
     wasabi is a good place to start .', 'label': '0'}
    Output:
    {'text': 'if you sometimes like to go to the movies to have fun ,
     wasabi is a good place to start .', 'label': 'positive'}
    """
    return [_replace_labels(dataset.info.features, sample) for sample in dataset]
