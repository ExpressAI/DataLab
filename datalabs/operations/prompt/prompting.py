from datalabs.operations.operation import text_operation, TextOperation


class Prompting(TextOperation):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(Prompting, self).__init__(*args, **kwargs)
        self._data_type = "TextData"


class prompting(text_operation):
    def __init__(self, *args, **kwargs):
        super(prompting, self).__init__(*args, **kwargs)

    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = Prompting(name=self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = Prompting(
                name=name,
                func=f,
                resources=self.resources,
                contributor=self.contributor,
                task=self.task,
                description=self.description,
            )
            return tf_cls
