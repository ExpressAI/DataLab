import os
import sys
import importlib
import inspect
import datalabs
import json
from datalabs import GeneratorBasedBuilder, DatasetBuilder

dir_datasets = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../datasets/",
)
sys.path.append(dir_datasets)





all_dataset_info = []
for file_name in os.listdir(dir_datasets):
    if not file_name.endswith(".py") and not file_name.endswith(".md") and not file_name.endswith(".pkl") and file_name!="__pycache__":
        # print(file_name)
        try:
            my_module = importlib.import_module(f"{file_name}.{file_name}")
        except:
            continue

        metadata = {}
        for name, obj in inspect.getmembers(my_module):


            if inspect.isclass(obj) and issubclass(obj, GeneratorBasedBuilder):
                if name == "Wmt":
                    continue


                data_info = obj()

                metadata["script_name"] = file_name
                metadata["dataset_class_name"] = name
                metadata["sub_datasets"] = list(data_info.builder_configs.keys())
                metadata["lanaguages"] = data_info._info().languages
                metadata["task_templates"] = None if data_info._info().task_templates is None else [x.task for x in data_info._info().task_templates]
                metadata["splits"] = data_info._info().splits



        all_dataset_info.append(metadata)


all_dataset_info_json = json.dumps(all_dataset_info, indent=4)
print(all_dataset_info_json)


