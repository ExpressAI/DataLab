from typing import Dict, List, Optional, Any
from typing import Callable, Mapping
from operation import TextOperation, text_operation
import os
import sys




class Prompting(TextOperation):

    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 task:str = None,
                 ):
        super().__init__(name, func, resources, contributor)
        self._type = "Prompting"
        self._data_type = "TextData"




class prompting(text_operation):
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


