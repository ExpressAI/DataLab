from typing import Any, Callable, Dict, List, Mapping, Optional

from operation import text_operation, TextOperation


class Editing(TextOperation):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(Editing, self).__init__(*args, **kwargs)
        self._data_type = "TextData"


class editing(text_operation):
    def __init__(self, *args, **kwargs):
        super(editing, self).__init__(*args, **kwargs)

    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = Editing(name=self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = Editing(
                name=name,
                func=f,
                resources=self.resources,
                contributor=self.contributor,
                task=self.task,
                description=self.description,
            )
            return tf_cls
