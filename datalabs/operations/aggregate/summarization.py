from typing import Any, Callable, Iterator, List, Mapping, Optional

import numpy as np
from tqdm import tqdm

from datalabs.operations.aggregate.aggregating import Aggregating, aggregating
from datalabs.operations.featurize import *  # noqa
from datalabs.operations.operation import dataset_operation, DatasetOperation


class SummarizationAggregating(Aggregating, DatasetOperation):
    def __init__(
        self,
        name: str = None,
        func: Callable[..., Any] = None,
        resources: Optional[Mapping[str, Any]] = None,
        contributor: str = None,
        processed_fields: List = ["text", "summary"],
        generated_field: str = None,
        task="summarization",
        description=None,
    ):
        super().__init__(
            name=name,
            func=func,
            resources=resources,
            contributor=contributor,
            task=task,
            description=description,
        )
        self._type = "SummarizationAggregating"
        self.processed_fields = ["text", "summary"]
        if isinstance(processed_fields, str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "SummarizationDataset"


class summarization_aggregating(aggregating, dataset_operation):
    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        contributor: str = None,
        processed_fields: List = ["text", "summary"],
        generated_field: str = None,
        task="summarization",
        description=None,
    ):
        super().__init__(
            name=name,
            resources=resources,
            contributor=contributor,
            description=description,
        )
        self.processed_fields = processed_fields
        self.generated_field = generated_field
        self.task = task

    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = SummarizationAggregating(name=self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = SummarizationAggregating(
                name=name,
                func=f,
                resources=self.resources,
                contributor=self.contributor,
                processed_fields=self.processed_fields,
                generated_field=self.generated_field,
                task=self.task,
                description=self.description,
            )
            return tf_cls


@summarization_aggregating(
    name="get_statistics",
    contributor="datalab",
    task="summarization",
    description="Calculate the overall statistics (e.g., density) "
    "of a given summarization dataset",
)
def get_statistics(samples: Iterator):
    """
        Input:
        samples: [{
         "text":
         "summary":
        }]
        Output:dict:

        usage:
        you can test it with following code:

    from datalabs import load_dataset
    from aggregate.summarization import *
    dataset = load_dataset('govreport')
    res = dataset['test'].apply(get_statistics)
    print(next(res))

    """

    summary_lengths = []
    text_lengths = []
    number_of_tokens = 0
    vocab = {}

    for sample in tqdm(samples):

        text, summary = sample["text"], sample["summary"]

        # average length of text
        text_length = len(text.split(" "))
        text_lengths.append(text_length)

        # average length of summary
        summary_length = len(summary.split(" "))
        summary_lengths.append(summary_length)

        # update the number of tokens
        number_of_tokens += len(text.split())
        number_of_tokens += len(summary.split())

        # Others
        # attr_json = get_all_features(sample)
        # attr_jsons.append(attr_json)

        # Vocabulary info
        for w in (text + summary).split(" "):

            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1

    res = {
        "dataset-level": {
            "average_text_length": np.average(text_lengths),
            "average_summary_length": np.average(summary_lengths),
            "length_info": {
                "max_text_length": np.max(text_lengths),
                "min_text_length": np.min(text_lengths),
                "average_text_length": np.average(text_lengths),
                "max_summary_length": np.max(summary_lengths),
                "min_summary_length": np.min(summary_lengths),
                "average_summary_length": np.average(summary_lengths),
            },
            "number_of_samples": len(samples),
            "number_of_tokens": number_of_tokens,
            # "vocabulary_info": vocab_sorted,
            # "gender_info": gender_ratio,
            # "hatespeech_info": hatespeech,
            # **attr_avg,
        },
        # "sample-level": sample_infos,
    }

    return res
