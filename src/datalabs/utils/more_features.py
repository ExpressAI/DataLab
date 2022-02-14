import string
import sys
from ..features.features import *


def prefix_dict_key(dict_obj, prefix):
    dict_obj_new = {}
    for k, v in dict_obj.items():
        dict_obj_new[prefix + "_" + k] = v
    return dict_obj_new

def get_feature_arguments(dict_output, field = "text", feature_level = "sample_level"):
    """Automate following code based on the output of `get_features_sample_level`
     additional_features = datalabs.Features(
        {
            TEXT+ "_" + "length": datalabs.Value(dtype="int64",
                                     is_bucket=True,
                                     ),
        }
    )
    """
    dict_feature_argument = {}
    for func_name, func_value in dict_output.items():
        key = field + "_" + func_name
        value = "int64"
        is_bucket = True
        if isinstance(func_value, int):
            value = "int64"
            is_bucket = True
        if isinstance(func_value, float):
            value = "float32"
            is_bucket = True
        elif isinstance(func_value, str):
            value = "string"
            is_bucket = True
        elif isinstance(func_value, dict):
            value = "dict"
            is_bucket = False

        if feature_level == "dataset_level":
            is_bucket = False
        #dict_feature_argument[key] = datalabs.Value(dtype=value, is_bucket=is_bucket, feature_level = feature_level, raw_feature = False)
        dict_feature_argument[key] = Value(dtype=value, is_bucket=is_bucket, feature_level = feature_level, raw_feature = False)

    return dict_feature_argument