from typing import Dict, List, Optional, Any
from typing import Callable, Mapping
from .prompting import Prompting, prompting
import os
import sys




class TextClassificationPrompting(Prompting):

    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 task:str = None,
                 ):
        super().__init__(name, func, resources, contributor)
        self._type = "TextClassificationPrompting"
        self._data_type = "Dataset"
        self.task = "text-classification"




class text_classification_prompting(prompting):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 task:str = None,
                 ):
        super().__init__(name, resources, contributor)
        self.task = task


    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = Prompting(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = Prompting(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                                 task = self.task)
            return tf_cls





"""
Input: huggingface dataset:
     dataset = load_dataset("ag_news") : dataset: DatasetDict
     
Output:
     List[{"text":text_new, "label":label_new}]     
     
     
Help:
>>> dataset['test'].features['label']
ClassLabel(num_classes=4, names=['World', 'Sports', 'Business', 'Sci/Tech'], names_file=None, id=None)     
"""




"""
Input:
    sample:dict: {"text":, "label_k"}
    tags_to_answers:    {"label_k": "answer_m"}
Output:
   
"""
@text_classification_prompting(name = "template1", contributor="datalab")
def template1(sample:dict, tags_to_answers:dict):


    text_new = sample["text"] + " What's this text about? " + " ".join(tags_to_answers.values())
    label_new = tags_to_answers[sample["label"]]


    return {"text":text_new, "label":label_new}







@text_classification_prompting(name = "template_1", contributor="datalab")
def get_copy_len(sample:dict):
    """
    text1
    text2
    label:values g()-> answers
    answers = g_{dataset}(label)
    """

    """
    Premise: <text> Hypothesis: <text2>
    target: True/False/Unknown
    """

    prompt = map(func, sample["text"])
    answer = map(func, sample["label"])

    return prompt, answer
