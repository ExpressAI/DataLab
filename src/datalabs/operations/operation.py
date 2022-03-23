from typing import Any, ClassVar, Dict, List, Optional
from typing import Callable, Mapping
import inspect


class OperationFunction:
    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor:str = None,
                 processed_fields = ["text"],
                 task = "Any",
                 description = None,
                 ):
        self.name = name
        self.func = func
        self.resources = resources or {}
        self.contributor = contributor
        self._type = self.__class__.__name__
        self.task = task

        self.processed_fields = ["text"]
        if isinstance(processed_fields,str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields

        # self.processed_fields = ["text"]
        self.generated_field = None
        self._data_type = self.__class__.__name__
        self.description = description


    def set(self, processed_fields):
        # print(self._type)
        return OperationFunction(name = self.name, func=self.func,
                                 resources=self.resources,
                                 contributor=self.contributor,
                                 description= self.description,
                                 processed_fields = processed_fields)

    def __call__(self, x: str) -> Any:  # str?
        """
        Parameters
        x: Text

        Returns
        Transformed Text
        """
        # return self.func(x, **self.resources)
        # print(inspect.getfullargspec(self.func))
        if "self" not in inspect.getfullargspec(self.func).args:
            return self.func(x, **self.resources)
        else: ## self.func is a member function of some class
            cls_obj = self.resources["cls"]
            del self.resources["cls"]
            return self.func(cls_obj, x, **self.resources)



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
    def __init__(self,*args,**kwargs,):
        super(TextOperation, self).__init__(*args, **kwargs)
        self._data_type = "TextData"


# x = TextOperation(name = "x")
# print(x._type)

class text_operation(operation_function):
    """

    """
    def __init__(self, *args,**kwargs):
        super(text_operation, self).__init__(*args, **kwargs)


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



# @text_operation(name="test",task="a",description="this is a test function")
# def get_sum(a,b):
#     return a+b
#
#
# print(get_sum.__dict__)



class StructuredTextOperation(TextOperation):
    def __init__(self,*args,**kwargs,):
        super(StructuredTextOperation, self).__init__(*args, **kwargs)
        self._data_type = "StructuredText"





class structured_text_operation(text_operation):
    """

    """
    def __init__(self, *args,**kwargs):
        super(structured_text_operation, self).__init__(*args, **kwargs)


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

    def __init__(self,*args,**kwargs,):
        super(DatasetOperation, self).__init__(*args, **kwargs)
        self._data_type = "Dataset"


class dataset_operation(text_operation):
    def __init__(self, *args,**kwargs):
        super(dataset_operation, self).__init__(*args, **kwargs)

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

