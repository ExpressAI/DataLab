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


@summarization_prompting(name="template_summ1", contributor="datalab",
                         template="{text} Write a TLDR (Too Long Didn\'\'t Read) summary for the above text.",
                         description="Prompt template: {text} Write a TLDR (Too Long Didn\'\'t Read) summary for the above text.",
                         task="summarization",
                         processed_fields=['text', 'summary'],)
def template_summ1(sample: dict):
    tp1 = "{text} Write a TLDR (Too Long Didn\\'t Read) summary for the above text."

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_summ2", contributor="datalab",
                         template="{text} Can you summarize the previous text?",
                         description="Prompt template: {text} Can you summarize the previous text?",
                         task="summarization",
                         processed_fields=['text', 'summary'],)
def template_summ2(sample: dict):
    tp1 = "{text} Can you summarize the previous text?"

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_summ3", contributor="datalab",
                         template="{text} what are the main points one should remember from this text?",
                         description="Prompt template: {text} what are the main points one should remember from this text?",
                         task="summarization",
                         processed_fields=['text', 'summary'],)
def template_summ3(sample: dict):
    tp1 = "{text} what are the main points one should remember from this text?"

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_summ4", contributor="datalab",
                         template="{text} In a few sentences, what does the previous paragraph say?",
                         description="Prompt template: {text} In a few sentences, what does the previous paragraph say?",
                         task="summarization",
                         processed_fields=['text', 'summary'],)
def template_summ4(sample: dict):
    tp1 = "{text} In a few sentences, what does the previous paragraph say?"

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_summ5", contributor="datalab",
                         template="{text} Condense the text down to the essentials.",
                         description="Prompt template: {text} Condense the text down to the essentials.",
                         task="summarization",
                         processed_fields=['text', 'summary'],)
def template_summ5(sample: dict):
    tp1 = "{text} Condense the text down to the essentials."

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_summ6", contributor="datalab",
                         template="{text} What can be a short description of the text?",
                         description="Prompt template: {text} What can be a short description of the text?",
                         task="summarization",
                         processed_fields=['text', 'summary'],)
def template_summ6(sample: dict):
    tp1 = "{text} What can be a short description of the text?"

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_summ7", contributor="datalab",
                         template="{text} How would you summarize the key points of the text?",
                         description="Prompt template: {text} How would you summarize the key points of the text?",
                         task="summarization",
                         processed_fields=['text', 'summary'],)
def template_summ7(sample: dict):
    tp1 = "{text} How would you summarize the key points of the text?"

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}


@summarization_prompting(name="template_summ8", contributor="datalab",
                         template="{text} Can you express the main content of the text?",
                         description="Prompt template: {text} Can you express the main content of the text?",
                         task="summarization",
                         processed_fields=['text', 'summary'],)
def template_summ8(sample: dict):
    tp1 = "{text} Can you express the main content of the text?"

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    return {"text_prompt": text_prompt,
            "summary_prompt": summary}
