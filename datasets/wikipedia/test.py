import os
import sys
import importlib
from featurize import WikipediaFeaturizing
from datalabs.operations.featurize.featurizing import featurizing, Featurizing



def get_operations(module_path:str):
    all_operations = {}
    module = importlib.import_module(module_path)

    target_class = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, Featurizing):
            #and "_data_type" in obj.__dict__.items():
            # print(obj.__dict__.items())
            target_class = obj
            break


    if target_class == None:
        raise ValueError("target class is none!")

    for name, obj in module.__dict__.items():
        if isinstance(obj, target_class):
            all_operations[obj] = name

    return all_operations




module_path = "featurize"
all_operations = get_operations(module_path)
print(all_operations)









# if isinstance(obj, type) and issubclass(obj, featurizing):
#     print(name)
#     print(obj)
#     print("------------------")