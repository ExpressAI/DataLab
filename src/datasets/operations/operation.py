from typing import Any, ClassVar, Dict, List, Optional
from typing import Callable, Mapping


class OperationFunction:
    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor:str = None,
                 processed_fields = None,
                 task = "Any",
                 description = None,
                 ):
        self.name = name
        self.func = func
        self.resources = resources or {}
        self.contributor = contributor
        self._type = "OperationFunction"
        self.target_filed = "sentence"
        self.task = task

        self.processed_fields = ["text"]
        if isinstance(processed_fields,str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields

        # self.processed_fields = ["text"]
        self.generated_field = None
        self._data_type = "Data"
        self.description = description


    def set(self, processed_fields):
        # print(self._type)
        return OperationFunction(name = self.name, func=self.func,
                                 resources=self.resources,
                                 contributor=self.contributor,
                                 description= self.description,
                                 processed_fields = processed_fields)
        # if isinstance(processed_field,str):
        #     cls(processed_fields = processed_field)
        # else:
        #     self.processed_fields = processed_field
        # return cls


    def __call__(self, x:str) -> Any: # str?
        """
        Parameters
        x: Text

        Returns
        Transformed Text
        """
        return self.func(x, **self.resources)



class operation_function:
    """

    """
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor:str = None,
                 task = "Any",
                 description = None,
                 ):
        self.name = name
        self.resources = resources or {}
        self.contributor = contributor
        self.task = task
        self.description = description


    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = OperationFunction(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            return OperationFunction(name = name, func=f, resources=self.resources, task = self.task,
                                     description=self.description)




class TextOperation(OperationFunction):
    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor:str = None,
                 task = "Any",
                 description = None,
                 ):
        super().__init__(name = name, func = func, resources =resources,
                         contributor=contributor, task=task,
                         description=description)
        self.name = name
        self.func = func
        self.resources = resources or {}
        self.contributor = contributor
        self._type = "TextOperation"
        self.target_filed = "sentence"
        self.processed_fields = ["text"]
        self.generated_field = None
        self._data_type = "TextData"
        self.task = task

    def __call__(self, x:str) -> Any: # str?
        """
        Parameters
        x: Text

        Returns
        Transformed Text
        """
        return self.func(x, **self.resources)



class text_operation(operation_function):
    """

    """
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor:str = None,
                 task = "Any",
                 description = None,
                 ):
        super().__init__(name = name, resources = resources, contributor = contributor,
                         task = task, description= description)

        self.name = name
        self.resources = resources or {}
        self.contributor = contributor
        self.task = task


    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = TextOperation(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            return TextOperation(name = name, func=f, resources=self.resources,
                                 task = self.task,
                                 description = self.description)


class StructuredTextOperation(TextOperation):
    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor:str = None,
                 task = "Any",
                 description = None,
                 ):
        super().__init__(name = name, func = func, resources =resources,
                         contributor=contributor, task=task,
                         description=description)
        self.name = name
        self.func = func
        self.resources = resources or {}
        self.contributor = contributor
        self._type = "StructuredTextOperation"
        self.target_filed = "structured_data"
        self.processed_fields = ["structured_data"]
        self.generated_field = None
        self._data_type = "StructuredText"
        self.task = task

    def __call__(self, x:str) -> Any: # str?
        """
        Parameters
        x: Text

        Returns
        Transformed Text
        """
        return self.func(x, **self.resources)



class structured_text_operation(text_operation):
    """

    """
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor:str = None,
                 task = "Any",
                 description = None,
                 ):
        super().__init__(name = name, resources = resources, contributor = contributor,
                         task = task, description= description)

        self.name = name
        self.resources = resources or {}
        self.contributor = contributor
        self.task = task


    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = StructuredTextOperation(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            return StructuredTextOperation(name = name, func=f, resources=self.resources,
                                 task = self.task,
                                 description = self.description)




class DatasetOperation(TextOperation):

    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 task = "Any",
                 description = None,
                 ):
        super().__init__(name, func, resources, contributor, task, description)
        self._type = "DatasetOperation"
        self._data_type = "Dataset"
        self.target_filed = "text"
        self.processed_fields = ["text"]
        self.generated_field = None
        self.task = task


class dataset_operation(text_operation):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 task = "Any",
                 description = None,
                 ):
        super().__init__(name = name, resources = resources, contributor = contributor, task = task, description= description)


    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = DatasetOperation(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = DatasetOperation(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                                   description=self.description)

            return tf_cls


