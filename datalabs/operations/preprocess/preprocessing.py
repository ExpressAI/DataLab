from datalabs.operations.operation import text_operation, TextOperation


class Preprocessing(TextOperation):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(Preprocessing, self).__init__(*args, **kwargs)
        self._data_type = "TextData"


class preprocessing(text_operation):
    def __init__(self, *args, **kwargs):
        super(preprocessing, self).__init__(*args, **kwargs)

    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = Preprocessing(name=self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = Preprocessing(
                name=name,
                func=f,
                resources=self.resources,
                contributor=self.contributor,
                task=self.task,
                description=self.description,
            )
            return tf_cls
