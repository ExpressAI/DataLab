from typing import Dict, List, Optional, Any
from typing import Callable, Mapping
from operation import TextOperation, text_operation


class Inference(TextOperation):

    def __init__(self, *args, **kwargs, ):
        super(Inference, self).__init__(*args, **kwargs)
        self._data_type = "TextData"


class inference(text_operation):

    def __init__(self, *args, **kwargs):
        super(inference, self).__init__(*args, **kwargs)

    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = Inference(name=self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = Inference(name=name, func=f,
                               resources=self.resources,
                               contributor=self.contributor,
                               task=self.task,
                               description=self.description, )
            return tf_cls
