from typing import Dict, List, Any, Optional
from typing import Callable, Mapping

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operation import DatasetOperation, dataset_operation
from prompt.prompting import *


class TopicClassificationPrompting(Prompting, DatasetOperation):

    def __init__(self,
                 name: str = None,
                 func: Callable[..., Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text", "label"],
                 generated_field: str = None,
                 task="topic-classification, text-classification",
                 description=None,
                 template=None,
                 ):
        super().__init__(name=name, func=func, resources=resources, contributor=contributor,
                         task=task, description=description)
        self._type = 'TopicClassificationPrompting'
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


class topic_classification_prompting(prompting, dataset_operation):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text", "label"],
                 generated_field: str = None,
                 task="topic-classification",
                 description=None,
                 template=None,
                 ):
        super().__init__(name=name, resources=resources, contributor=contributor, description=description)
        self.processed_fields = processed_fields
        self.generated_field = generated_field
        self.task = task
        self.template = template

    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = TopicClassificationPrompting(name=self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = TopicClassificationPrompting(name=name, func=f,
                                                  resources=self.resources,
                                                  contributor=self.contributor,
                                                  processed_fields=self.processed_fields,
                                                  generated_field=self.generated_field,
                                                  task=self.task,
                                                  description=self.description,
                                                  template=self.template, )
            return tf_cls


"""
>>> labels = [positive, negative]
>>> Topic = "I love this movie"
>>> texture_choice = ", ".join(labels[:-1]) + " or " + labels[-1] + "?"
>>> template = f"Given the text {text}, is it {texture_choice}">>> template
'Given the text I love this movie, is it positive or negative?';


Test Example:

from datalabs import load_dataset
dataset = load_dataset('ag_news')
from prompt.topic_classification import *
res = dataset['test'].apply(template_p1)
print(next(res))



"""


@topic_classification_prompting(name="template_tc1", contributor="datalab", task="topic-classification, text-classification",
                                description="Prompt template: Given the text: {text}, is it about {texture_choices}",
                                template = "Given the text: {text}, is it about {texture_choices}",
                                processed_fields=['text', 'label'],
                                )
def template_tc1(sample: dict, labels_to_answers: Dict):
    tp1 = "Given the text: {text}, is it about {texture_choices}"

    # prompting process
    answers = list(labels_to_answers.values())
    text = sample["text"]
    texture_choices = ", ".join(answers[:-1]) + " or " + answers[-1] + "?"

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@topic_classification_prompting(name="template_tc2", contributor="datalab",
                                template="Given the text: {text}, it is about [mask]",
                                description="Prompt template: Given the text: {text}, it is about [mask]",
                                task="topic-classification, text-classification",
                                processed_fields=['text', 'label'],)
def template_tc2(sample: dict, labels_to_answers: Dict):
    tp = "Given the text: {text}, it is about [mask]"

    # prompting process
    text = sample["text"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@topic_classification_prompting(name="template_tc3", contributor="datalab",
                                template="Given the text: {text} Classify this text. You may choose from {texture_choices}.",
                                description ="Prompt template: Given the text: {text} Classify this text. You may choose from {texture_choices}.",
                                task="topic-classification, text-classification",
                                processed_fields=['text', 'label'],)
def template_tc3(sample: dict, labels_to_answers: Dict):
    tp = "Given the text: {text} Classify this text. You may choose from {texture_choices}."

    # prompting process
    text = sample["text"]
    answers = list(labels_to_answers.values())
    texture_choices = ", ".join(answers)

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@topic_classification_prompting(name="template_tc4", contributor="datalab",
                                template = "Given the text: {text} Given a list of categories: {texture_choices}, what category does the paragraph belong to?",
                                description = "Prompt template: Given the text: {text} Given a list of categories: {texture_choices}, what category does the paragraph belong to?",
                                task="topic-classification, text-classification",
                                processed_fields=['text', 'label'],)
def template_tc4(sample: dict, labels_to_answers: Dict):
    tp = "Given the text: {text} Given a list of categories: {texture_choices}, what category does the paragraph belong to?"

    # prompting process
    text = sample["text"]
    answers = list(labels_to_answers.values())
    texture_choices = ", ".join(answers)

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@topic_classification_prompting(name="template_tc5", contributor="datalab",
                                template="Given the text: {text} Pick one category for the previous text. The options are {texture_choices}.",
                                description="Prompt template: Given the text: {text} Pick one category for the previous text. The options are {texture_choices}.",
                                task="topic-classification, text-classification",
                                processed_fields=['text', 'label'],)
def template_tc5(sample: dict, labels_to_answers: Dict):
    tp = "Given the text: {text} Pick one category for the previous text. The options are {texture_choices}."

    # prompting process
    text = sample["text"]
    answers = list(labels_to_answers.values())
    texture_choices = ", ".join(answers)

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@topic_classification_prompting(name="template_tc6", contributor="datalab",
                                template="Given the text: {text} Pick one category for the previous text. The options are {texture_choices}.",
                                description="Prompt template: Given the text: {text} Pick one category for the previous text. The options are {texture_choices}.",
                                task="topic-classification, text-classification",
                                processed_fields=['text', 'label'],)
def template_tc6(sample: dict, labels_to_answers: Dict):
    tp = "Given the text: {text} Can you identify the category of this text? {texture_choices}"

    # prompting process
    text = sample["text"]
    answers = list(labels_to_answers.values())
    texture_choices = ", ".join(answers[:-1]) + " or " + answers[-1] + "?"

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@topic_classification_prompting(name="template_tc7", contributor="datalab",
                                template="Given the text: {text} What's the main topic of this paragraph? {texture_choices}",
                                description="Prompt template: Prompt template: Given the text: {text} What's the main topic of this paragraph? {texture_choices}",
                                task="topic-classification, text-classification",
                                processed_fields=['text', 'label'],)
def template_tc7(sample: dict, labels_to_answers: Dict):
    tp = "Given the text: {text} What\\'s the main topic of this paragraph? {texture_choices}"

    # prompting process
    text = sample["text"]
    answers = list(labels_to_answers.values())
    texture_choices = ", ".join(answers[:-1]) + " or " + answers[-1] + "?"

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@topic_classification_prompting(name="template_tc8", contributor="datalab",
                                template="Given the text: {text} Is this a piece of text regarding {texture_choices}",
                                description="Prompt template: Prompt template: Given the text: {text} Is this a piece of text regarding {texture_choices}",
                                task="topic-classification, text-classification",
                                processed_fields=['text', 'label'],)
def template_tc8(sample: dict, labels_to_answers: Dict):
    tp = "Given the text: {text} Is this a piece of text regarding {texture_choices}"

    # prompting process
    text = sample["text"]
    answers = list(labels_to_answers.values())
    texture_choices = ", ".join(answers[:-1]) + " or " + answers[-1] + "?"

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}
