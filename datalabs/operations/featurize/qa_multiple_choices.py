from typing import Any, Callable, List, Mapping, Optional

from datalabs.operations.featurize.featurizing import Featurizing, featurizing
from datalabs.operations.featurize.general import (
    get_features_sample_level as get_features_sample_level_general,
)
from datalabs.operations.operation import dataset_operation, DatasetOperation


class QuestionAnsweringMultipleChoicesFeaturizing(Featurizing, DatasetOperation):
    def __init__(
        self,
        name: str = None,
        func: Callable[..., Any] = None,
        resources: Optional[Mapping[str, Any]] = None,
        contributor: str = None,
        processed_fields: List = ["text"],
        generated_field: str = None,
        task="question-answering-multiple-choices",
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
        self._type = "QuestionAnsweringMultipleChoicesFeaturizing"
        self.processed_fields = ["text"]
        if isinstance(processed_fields, str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "Dataset"


class qa_multiple_choices_featurizing(featurizing, dataset_operation):
    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        contributor: str = None,
        processed_fields: List = ["text"],
        generated_field: str = None,
        task="question-answering-multiple-choices",
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
            tf_class = QuestionAnsweringMultipleChoicesFeaturizing(
                name=self.name.__name__, func=self.name
            )
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = QuestionAnsweringMultipleChoicesFeaturizing(
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


def get_schema_of_sample_level_features():
    return {
        "context_length": 1,
        "context_lexical_richness": 0.2,
        "context_basic_words": 0.2,
        "context_gender_bias_word_male": 1,
        "context_gender_bias_word_female": 2,
        "context_gender_bias_single_name_male": 1,
        "context_gender_bias_single_name_female": 1,
        "answer_length": 1,
        "option1_length": 1,
        "option2_length": 1,
        # "option_index": 0,
    }


@qa_multiple_choices_featurizing(
    name="get_features_sample_level",
    contributor="datalab",
    processed_fields="text",
    task="question-answering-multiple-choices",
    description="This function is used to calculate the text length",
)
def get_features_sample_level(sample: dict):

    # print(sample)
    context = sample["context"]
    options = sample["options"]
    answer = sample["answers"]["text"]

    answer_length = len(answer.split(" "))
    option1_length = len(options[0].split(" "))
    option2_length = len(options[1].split(" "))

    res_info_general = get_features_sample_level_general.func(context)
    res_info_general_new = {}
    for k, v in res_info_general.items():
        res_info_general_new["context" + "_" + k] = v

    res_info_general_new.update(
        {
            "answer_length": answer_length,
            "option1_length": option1_length,
            "option2_length": option2_length,
            # "option_index":int(option_index),
        }
    )

    return res_info_general_new
