from typing import Dict, List, Any, Optional
from typing import Callable, Mapping

import os
import sys  # contradiction, entailment, neutral

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operation import DatasetOperation, dataset_operation
from prompt.prompting import *


class NLIPrompting(Prompting, DatasetOperation):

    def __init__(self,
                 name: str = None,
                 func: Callable[..., Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text", "label"],
                 generated_field: str = None,
                 task="natural-language-inference",
                 description=None,
                 template=None,
                 ):
        super().__init__(name=name, func=func, resources=resources, contributor=contributor,
                         task=task, description=description)
        self._type = 'NLIPrompting'
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


class nli_prompting(prompting, dataset_operation):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text", "label"],
                 generated_field: str = None,
                 task="natural-language-inference",
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
            tf_class = NLIPrompting(name=self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = NLIPrompting(name=name, func=f,
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
>>> text = "I love this movie"
>>> texture_choice = ", ".join(labels[:-1]) + " or " + labels[-1] + "?"
>>> template = f"Given the text {text}, is it {texture_choice}">>> template
'Given the text I love this movie, is it positive or negative?';


Test Example:

from datalabs import load_dataset
dataset = load_dataset('sick')
from prompt.natural_language_inference import *
res = dataset['test'].apply(template_p1)
print(next(res))



"""


@nli_prompting(name="template_p1", contributor="datalab", processed_fields=['text1', 'text2', 'label'],
               template="Given that \"{text1}\" Can we infer that \"{text2}\"? Yes or No or Unknown?",
               task="natural-language-inference")
def template_p1(sample: dict, labels_to_answers: Dict):
    # labels=('contradiction', 'entailment', 'neutral'))
    answers_to_desc = {'contradiction': "No",
                       'entailment': "Yes",
                       'neutral': 'Unknown'}

    tp1 = "Given that \"{text1}\" Can we infer that {text2}? Yes or No or Unknown?"

    # prompting process
    answers = list(labels_to_answers.values())
    text1 = sample["text1"]
    text2 = sample["text2"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    label_prompt = answers_to_desc[labels_to_answers[sample["label"]]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@nli_prompting(name="template_p2", contributor="datalab", processed_fields=['text1', 'text2', 'label'],
               template="Given text {text1} and text {text2}, is their relationship {texture_choices}",
               task="natural-language-inference")
def template_p2(sample: dict, labels_to_answers: Dict):
    tp = "Given text: {text1} and text: {text2}, is their relationship {texture_choices}"

    # prompting process
    text1 = sample["text1"]
    text2 = sample["text2"]
    answers = list(labels_to_answers.values())
    texture_choices = ", ".join(answers[:-1]) + " or " + answers[-1] + "?"

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@nli_prompting(name="template_p3", contributor="datalab", processed_fields=['text1', 'text2', 'label'],
               template="The relationship of two texts {text1} and {text2} is [mask]",
               task="natural-language-inference")
def template_p3(sample: dict, labels_to_answers: Dict):
    # labels=('contradiction', 'entailment', 'neutral'))

    tp1 = "The relationship of two texts {text1} and {text2} is [mask]"

    # prompting process

    text1 = sample["text1"]
    text2 = sample["text2"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@nli_prompting(name="template_p4", contributor="datalab", processed_fields=['text1', 'text2', 'label'],
               template="{text1} {text2} True or False or Unknown?",
               task="natural-language-inference")
def template_p4(sample: dict, labels_to_answers: Dict):
    answers_to_desc = {'contradiction': "False",
                       'entailment': "True",
                       'neutral': 'Unknown'}
    tp = "{text1} {text2} True or False or Unknown?"
    text1 = sample["text1"]
    text2 = sample["text2"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp))
    label_prompt = answers_to_desc[labels_to_answers[sample["label"]]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@nli_prompting(name="template_p5", contributor="datalab", processed_fields=['text1', 'text2', 'label'],
               template="{text1} Is the following statement True or False or Unknown: {text2}?",
               task="natural-language-inference")
def template_p5(sample: dict, labels_to_answers: Dict):
    answers_to_desc = {'contradiction': "False",
                       'entailment': "True",
                       'neutral': 'Unknown'}
    tp = "{text1} Is the following statement True or False or Unknown: {text2}?"
    text1 = sample["text1"]
    text2 = sample["text2"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp))
    label_prompt = answers_to_desc[labels_to_answers[sample["label"]]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@nli_prompting(name="template_p6", contributor="datalab", processed_fields=['text1', 'text2', 'label'],
               template="Premise: {text1} Hypothesis: {text2} Based on the premise, is the hypothesis true or false or undetermined?",
               task="natural-language-inference")
def template_p6(sample: dict, labels_to_answers: Dict):
    answers_to_desc = {'contradiction': "false",
                       'entailment': "true",
                       'neutral': 'undetermined'}
    tp = "Premise: {text1} Hypothesis: {text2} Based on the premise, is the hypothesis true or false or undetermined?"
    text1 = sample["text1"]
    text2 = sample["text2"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp))
    label_prompt = answers_to_desc[labels_to_answers[sample["label"]]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@nli_prompting(name="template_p7", contributor="datalab", processed_fields=['text1', 'text2', 'label'],
               template="Premise: {text1} Hypothesis: {text2} The relation between the hypothesis and premise is [mask]",
               task="natural-language-inference")
def template_p7(sample: dict, labels_to_answers: Dict):
    tp = "Premise: {text1} Hypothesis: {text2} The relation between the hypothesis and premise is [mask]"
    text1 = sample["text1"]
    text2 = sample["text2"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp))
    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@nli_prompting(name="template_p7", contributor="datalab", processed_fields=['text1', 'text2', 'label'],
               template="Premise: {text1} Hypothesis: {text2} The relation between the hypothesis and premise is [mask]",
               task="natural-language-inference")
def template_p7(sample: dict, labels_to_answers: Dict):
    tp = "Premise: {text1} Hypothesis: {text2} The relation between the hypothesis and premise is [mask]"
    text1 = sample["text1"]
    text2 = sample["text2"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp))
    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@nli_prompting(name="template_p8", contributor="datalab", processed_fields=['text1', 'text2', 'label'],
               template="{text1} Based on that information, is the claim \"{text2}\" true, false or inconclusive?",
               task="natural-language-inference")
def template_p8(sample: dict, labels_to_answers: Dict):
    answers_to_desc = {'contradiction': "false",
                       'entailment': "true",
                       'neutral': 'inconclusive'}
    tp = "{text1} Based on that information, is the claim \"{text2}\" true, false or inconclusive?"
    text1 = sample["text1"]
    text2 = sample["text2"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp))
    label_prompt = answers_to_desc[labels_to_answers[sample["label"]]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@nli_prompting(name="template_p9", contributor="datalab", processed_fields=['text1', 'text2', 'label'],
               template="{text1} Does it imply that \"{text2}\"? Yes, No or Maybe?",
               task="natural-language-inference")
def template_p9(sample: dict, labels_to_answers: Dict):
    answers_to_desc = {'contradiction': "No",
                       'entailment': "Yes",
                       'neutral': 'Maybe'}
    tp = "{text1} Does it imply that \"{text2}\"? Yes, No or Maybe?"
    text1 = sample["text1"]
    text2 = sample["text2"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp))
    label_prompt = answers_to_desc[labels_to_answers[sample["label"]]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}


@nli_prompting(name="template_p10", contributor="datalab", processed_fields=['text1', 'text2', 'label'],
               template="Assume it is true that {text1}. Therefore, {text2} is guaranteed, possible or impossible?",
               task="natural-language-inference")
def template_p10(sample: dict, labels_to_answers: Dict):
    answers_to_desc = {'contradiction': "impossible",
                       'entailment': "guaranteed",
                       'neutral': 'possible'}
    tp = "Assume it is true that {text1}. Therefore, {text2} is guaranteed, possible or impossible?"
    text1 = sample["text1"]
    text2 = sample["text2"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp))
    label_prompt = answers_to_desc[labels_to_answers[sample["label"]]]

    return {"text_prompt": text_prompt,
            "label_prompt": label_prompt}
