from typing import Any, Callable, List, Mapping, Optional

from datalabs.operations.featurize.featurizing import Featurizing, featurizing
from datalabs.operations.operation import dataset_operation, DatasetOperation


class NLPFeaturizing(Featurizing, DatasetOperation):
    def __init__(
        self,
        name: str = None,
        func: Callable[..., Any] = None,
        resources: Optional[Mapping[str, Any]] = None,
        contributor: str = None,
        processed_fields: List = ["text"],
        generated_field: str = None,
        task="text-matching",
        description=None,
    ):
        super().__init__(
            name=name,
            func=func,
            resources=resources,
            contributor=contributor,
            task=task,
            description=description,
        )
        self._type = "NLPFeaturizing"
        self.processed_fields = ["text"]
        if isinstance(processed_fields, str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "Dataset"


class nlp_featurizing(featurizing, dataset_operation):
    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        contributor: str = None,
        processed_fields: List = ["text"],
        generated_field: str = None,
        task="NLP",
        description=None,
    ):
        super().__init__(
            name=name,
            resources=resources,
            contributor=contributor,
            description=description,
        )
        self.processed_fields = processed_fields
        self.generated_field = generated_field
        self.task = task

    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = NLPFeaturizing(name=self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = NLPFeaturizing(
                name=name,
                func=f,
                resources=self.resources,
                contributor=self.contributor,
                processed_fields=self.processed_fields,
                generated_field=self.generated_field,
                task=self.task,
                description=self.description,
            )
            return tf_cls
