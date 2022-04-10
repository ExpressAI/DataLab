from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Callable
import os
import sys
from typing import Any, Optional


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from operation import DatasetOperation, dataset_operation
from prompt.prompting import *


class TextClassificationPrompting(Prompting, DatasetOperation):
    def __init__(
        self,
        name: str = None,
        func: Callable[..., Any] = None,
        resources: Optional[Mapping[str, Any]] = None,
        contributor: str = None,
        processed_fields: list = ["text", "label"],
        generated_field: str = None,
        task="text-classification",
        description=None,
        template=None,
    ):
        super().__init__(
            name=name,
            func=func,
            resources=resources,
            contributor=contributor,
            task=task,
            description=description,
        )
        self._type = "TextClassificationPrompting"
        self.processed_fields = ["text", "label"]
        if isinstance(processed_fields, str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "Dataset"
        self.template = template

    def __call__(self, sample, labels_to_answers) -> Any:  # str?
        """
        Parameters
        x: Text

        Returns
        Transformed Text
        """
        return self.func(sample, labels_to_answers, **self.resources)


class text_classification_prompting(prompting, dataset_operation):
    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        contributor: str = None,
        processed_fields: list = ["text", "label"],
        generated_field: str = None,
        task="text-classification",
        description=None,
        template=None,
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
        self.template = template

    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = TextClassificationPrompting(
                name=self.name.__name__, func=self.name
            )
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = TextClassificationPrompting(
                name=name,
                func=f,
                resources=self.resources,
                contributor=self.contributor,
                processed_fields=self.processed_fields,
                generated_field=self.generated_field,
                task=self.task,
                description=self.description,
                template=self.template,
            )
            return tf_cls


"""
>>> labels = [positive, negative]
>>> text = "I love this movie"
>>> texture_choice = ", ".join(labels[:-1]) + " or " + labels[-1] + "?"
>>> template = f"Given the text {text}, is it {texture_choice}">>> template
'Given the text I love this movie, is it positive or negative?';


Test Example:

from datalabs import load_dataset
dataset = load_dataset('mr')
from prompt.text_classification import *
res = dataset['test'].apply(template_p1)
print(next(res))



"""


@text_classification_prompting(
    name="template_p1",
    contributor="datalab",
    processed_fields=["text", "label"],
    template="Given the text: {text}, is it about {texture_choices}",
    task="text-classification",
    description="this function is used to calculate the text length",
)
def template_p1(sample: dict, labels_to_answers: dict):

    tp1 = "Given the text: {text}, is it about {texture_choices}"

    # prompting process
    answers = list(labels_to_answers.values())
    text = sample["text"]
    texture_choices = ", ".join(answers[:-1]) + " or " + answers[-1] + "?"

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt, "label_prompt": label_prompt}


@text_classification_prompting(
    name="template_p2",
    contributor="datalab",
    processed_fields=["text", "label"],
    template="Given the text: {text}, it is about [mask]",
    task="text-classification",
    description="this function is used to calculate the text length",
)
def template_p2(sample: dict, labels_to_answers: dict):

    tp = "Given the text: {text}, it is about [mask]"

    # prompting process
    text = sample["text"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt, "label_prompt": label_prompt}


# @text_classification_prompting(name = "template_p1", contributor= "datalab", processed_fields= ['text', 'label'],
#                                  task="text-classification", description="this function is used to calculate the text length")
# def template_p1(sample:dict, labels_to_answers:Dict, template:str):
#
#     template = "Given the text: {text}, is it about {all_answer_options}"
#
#     # prompting process
#     answers = list(labels_to_answers.values())
#     text = sample["text"]
#     all_answer_options = ", ".join(answers[:-1]) + " or " + answers[-1] + "?"
#
#     # instantiation
#     text_prompt = eval("f'{}'".format(template))
#
#     label_prompt = labels_to_answers[sample["label"]]
#
#     return {"text_prompt": text_prompt,
#             "label_prompt": label_prompt}
