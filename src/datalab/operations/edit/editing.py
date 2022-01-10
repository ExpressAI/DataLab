from typing import Dict, List, Optional, Any
from typing import Callable, Mapping
from operation import TextOperation, text_operation





class Editing(TextOperation):

    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 task = "Any",
                 description = None,
                 ):
        super().__init__(name, func, resources, contributor,
                         task = task, description=description,)
        self._type = "Editing"
        self._data_type = "TextData"
        self.processed_fields = ["text"]



class editing(text_operation):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 task="Any",
                 description=None,
                 ):
        super().__init__(name, resources, contributor,
                         task = task, description=description,
                         )


    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = Editing(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = Editing(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                             task = self.task, description=self.description)
            return tf_cls



