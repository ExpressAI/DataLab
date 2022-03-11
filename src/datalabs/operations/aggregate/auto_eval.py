from typing import Dict, List, Any, Optional
from .aggregating import Aggregating, aggregating
from typing import Callable, Mapping, Iterator
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operation import DatasetOperation, dataset_operation
from data import TextData


# Build-in ExplainaBoard
# from datalabs import get_processor
# from datalabs import get_loader
# from datalabs.tasks.task_info import get_task_categories, TaskType
# from datalabs.constants import FileType, Source

# Build-out ExplainaBoard
# from explainaboard import get_processor
# from explainaboard import get_loader
# from explainaboard import get_task_categories, TaskType
# from explainaboard import FileType, Source



class AutoEval(Aggregating, DatasetOperation):


    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text"],
                 generated_field: str = None,
                 task = "text-classification",
                 description = None,
                 ):
        super().__init__(name = name, func = func, resources = resources, contributor = contributor,
                         task = task,description=description)
        self._type = 'AutoEval'
        self.processed_fields = ["text"]
        if isinstance(processed_fields,str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "Dataset"




class auto_eval(aggregating, dataset_operation):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text"],
                 generated_field:str = None,
                 task = "text-classification",
                 description = None,
                 ):
        super().__init__(name = name, resources = resources, contributor = contributor, description=description)
        self.processed_fields = processed_fields
        self.generated_field = generated_field
        self.task = task


    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = AutoEval(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = AutoEval(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                                    processed_fields = self.processed_fields,
                                    generated_field = self.generated_field,
                                    task = self.task,
                                    description=self.description,)
            return tf_cls


# @auto_eval(name="explainaboard")
# def explainaboard(samples:Iterator, dataset_info = None):
#     """
#     Package: python
#     Input:
#         texts: Iterator
#     Output:
#         int
#     """
#
#     # Setup metadata
#     metadata = {
#         "dataset_name": dataset_info.builder_name,
#         "sub_dataset_name": dataset_info.config_name,
#         "task_name": dataset_info.task_templates[0].task_category,
#         "reload_stat":True,
#                 }
#
#     # if metric_names is not None:
#     #     metadata["metric_names"] = metric_names
#
#     loader = get_loader(
#         dataset_info.task_templates[0].task_category,
#         Source.in_memory,
#         FileType.datalab,
#         samples,
#     )
#
#     data = loader.load()
#
#
#     # Run analysis
#     report = get_processor(dataset_info.task_templates[0].task_category, metadata=metadata, data=data).process()
#
#
#
#
#     return {"report":report}
