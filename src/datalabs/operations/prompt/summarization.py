from typing import Dict, List, Any, Optional
from typing import Callable, Mapping

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operation import DatasetOperation, dataset_operation
from prompt.prompting import *


class SummarizationPrompting(Prompting, DatasetOperation):

    def __init__(self,
                 name: str = None,
                 func: Callable[..., Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text", "summary"],
                 generated_field: str = None,
                 task="summarization",
                 description=None,
                 template=None,
                 ):
        super().__init__(name=name, func=func, resources=resources, contributor=contributor,
                         task=task, description=description)
        self._type = 'SummarizationPrompting'
        self.processed_fields = ["text", "summary"]
        if isinstance(processed_fields, str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "Dataset"
        self.template = template

    def __call__(self, sample) -> Any:  # str?
        """
        Parameters
        x: Text

        Returns
        Transformed Text
        """
        return self.func(sample, **self.resources)


class summarization_prompting(prompting, dataset_operation):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text", "summary"],
                 generated_field: str = None,
                 task="summarization",
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
            tf_class = SummarizationPrompting(name=self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = SummarizationPrompting(name=name, func=f,
                                            resources=self.resources,
                                            contributor=self.contributor,
                                            processed_fields=self.processed_fields,
                                            generated_field=self.generated_field,
                                            task=self.task,
                                            description=self.description,
                                            template=self.template, )
            return tf_cls


"""
 


Test Example:

from datalabs import load_dataset
dataset = load_dataset('xsum')
from prompt.summarization import *
res = dataset['test'].apply(template_p1)
print(next(res))



"""


@summarization_prompting(name="template_p1", contributor="datalab", processed_fields=['text', 'summary'],
                         template="{text} Write a TLDR (Too Long Didn\'\'t Read) summary for the above text.",
                         task="summarization")
def template_p1(sample: dict):
    tp1 = "{text} Write a TLDR (Too Long Didn\\'t Read) summary for the above text."

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_p2", contributor="datalab", processed_fields=['text', 'summary'],
                         template="{text} Can you summarize the previous text?",
                         task="summarization")
def template_p2(sample: dict):
    tp1 = "{text} Can you summarize the previous text?"

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_p3", contributor="datalab", processed_fields=['text', 'summary'],
                         template="{text} what are the main points one should remember from this text?",
                         task="summarization")
def template_p3(sample: dict):
    tp1 = "{text} what are the main points one should remember from this text?"

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_p4", contributor="datalab", processed_fields=['text', 'summary'],
                         template="{text} In a few sentences, what does the previous paragraph say?",
                         task="summarization")
def template_p4(sample: dict):
    tp1 = "{text} In a few sentences, what does the previous paragraph say?"

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_p5", contributor="datalab", processed_fields=['text', 'summary'],
                         template="{text} Condense the text down to the essentials.",
                         task="summarization")
def template_p5(sample: dict):
    tp1 = "{text} Condense the text down to the essentials."

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_p6", contributor="datalab", processed_fields=['text', 'summary'],
                         template="{text} What can be a short description of the text?",
                         task="summarization")
def template_p6(sample: dict):
    tp1 = "{text} What can be a short description of the text?"

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_p7", contributor="datalab", processed_fields=['text', 'summary'],
                         template="{text} How would you summarize the key points of the text?",
                         task="summarization")
def template_p7(sample: dict):
    tp1 = "{text} How would you summarize the key points of the text?"

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_p8", contributor="datalab", processed_fields=['text', 'summary'],
                         template="{text} Can you express the main content of the text?",
                         task="summarization")
def template_p8(sample: dict):
    tp1 = "{text} Can you express the main content of the text?"

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}