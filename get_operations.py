import ast
import importlib
import inspect
from inspect import getmembers, isfunction
import json
import os
import sys
from textwrap import dedent

from featurize import general

import datalabs

# print(general)
#
# a = [m[0] for m in inspect.getmembers(general, inspect.isfunction) if m[1].__module__ == 'general']
# print(a)
# func_list = getmembers(general, isfunction)
#
# for func in func_list:
#     print(inspect.getsource(func[1]))


# https://stackoverflow.com/questions/5910703/how-to-get-all-methods-of-a-python-class-with-given-decorator
def findDecorators(target):
    import ast
    import inspect

    res = {}

    def visit_FunctionDef(node):
        res[node.name] = [ast.dump(e) for e in node.decorator_list]

    V = ast.NodeVisitor()
    V.visit_FunctionDef = visit_FunctionDef
    V.visit(compile(inspect.getsource(target), "?", "exec", ast.PyCF_ONLY_AST))
    return res


def parse_dec_ast(info):
    dec_class_type = info.split("func=Name(id='")[-1].split("',")[0]
    if dec_class_type.find("_") != -1:
        dec_class_type = dec_class_type.split("_")[-1]

    arg_keys = []
    for x in info.split("keyword(arg='")[1:]:
        arg_keys.append(x.split("', value=")[0])

    arg_values = []
    for x in info.split("value=Str(s='")[1:]:
        arg_values.append(x.split("'))")[0])

    new_dict = dict(zip(arg_keys, arg_values))
    res = {"class_type": dec_class_type, "args": new_dict}

    return res


ALL_FUNCS = []


"""
featurize.general
"""
from featurize import general

funcs = findDecorators(general)
for k, v in funcs.items():
    if len(v) == 0:
        continue
    info = v[0]
    func_metadata = parse_dec_ast(info)
    ALL_FUNCS.append(func_metadata)

# featurize.summarization
from featurize import summarization

funcs = findDecorators(summarization)
for k, v in funcs.items():
    if len(v) == 0:
        continue
    info = v[0]
    func_metadata = parse_dec_ast(info)
    ALL_FUNCS.append(func_metadata)


"""
edit.general
"""
from edit import general

funcs = findDecorators(general)
for k, v in funcs.items():
    if len(v) == 0:
        continue
    info = v[0]
    func_metadata = parse_dec_ast(info)
    ALL_FUNCS.append(func_metadata)


"""
edit.general.plugins
"""
dir_operations = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src/datalabs/operations/edit/plugins/general/",
)
sys.path.append(dir_operations)

# print(dir_operations)
for file_name in os.listdir(dir_operations):
    if not file_name.endswith(".py") and file_name != "__pycache__":
        # print(f"{file_name}.transformation.{file_name}")
        my_module = importlib.import_module(f"{file_name}.transformation")

        # extract metadata information given the module: "my_module:
        funcs = findDecorators(my_module)
        for k, v in funcs.items():
            if len(v) == 0:
                continue
            info = v[0]
            func_metadata = parse_dec_ast(info)
            ALL_FUNCS.append(func_metadata)


"""
preprocess.general
"""

from preprocess import general

funcs = findDecorators(general)
for k, v in funcs.items():
    if len(v) == 0:
        continue
    info = v[0]
    func_metadata = parse_dec_ast(info)
    ALL_FUNCS.append(func_metadata)


"""
aggregate.general
"""
from aggregate import general

funcs = findDecorators(general)
for k, v in funcs.items():
    if len(v) == 0:
        continue
    info = v[0]
    func_metadata = parse_dec_ast(info)
    ALL_FUNCS.append(func_metadata)


# aggregate.summarization
from aggregate import summarization

funcs = findDecorators(summarization)
for k, v in funcs.items():
    if len(v) == 0:
        continue
    info = v[0]
    func_metadata = parse_dec_ast(info)
    ALL_FUNCS.append(func_metadata)


# aggregate.sequence_labeling
from aggregate import sequence_labeling

funcs = findDecorators(sequence_labeling)
for k, v in funcs.items():
    if len(v) == 0:
        continue
    info = v[0]
    func_metadata = parse_dec_ast(info)
    ALL_FUNCS.append(func_metadata)


# aggregate.text_matching
from aggregate import text_matching

funcs = findDecorators(text_matching)
for k, v in funcs.items():
    if len(v) == 0:
        continue
    info = v[0]
    func_metadata = parse_dec_ast(info)
    ALL_FUNCS.append(func_metadata)


# aggregate.text_classification
from aggregate import text_classification

funcs = findDecorators(text_classification)
for k, v in funcs.items():
    if len(v) == 0:
        continue
    info = v[0]
    func_metadata = parse_dec_ast(info)
    ALL_FUNCS.append(func_metadata)


# ------------------------- prompt ------------------
# aggregate.sequence_labeling
from prompt import topic_classification

funcs = findDecorators(topic_classification)
for k, v in funcs.items():
    if len(v) == 0:
        continue
    info = v[0]
    func_metadata = parse_dec_ast(info)
    print(func_metadata)
    ALL_FUNCS.append(func_metadata)


from prompt import summarization

funcs = findDecorators(summarization)
for k, v in funcs.items():
    if len(v) == 0:
        continue
    info = v[0]
    func_metadata = parse_dec_ast(info)
    print(func_metadata)
    ALL_FUNCS.append(func_metadata)


from prompt import sentiment_classification

funcs = findDecorators(sentiment_classification)
for k, v in funcs.items():
    if len(v) == 0:
        continue
    info = v[0]
    func_metadata = parse_dec_ast(info)
    print(func_metadata)
    ALL_FUNCS.append(func_metadata)


from prompt import natural_language_inference

funcs = findDecorators(natural_language_inference)
for k, v in funcs.items():
    if len(v) == 0:
        continue
    info = v[0]
    func_metadata = parse_dec_ast(info)
    print(func_metadata)
    ALL_FUNCS.append(func_metadata)


for k in ALL_FUNCS:
    print(k)


with open("./docs/Resources/operations_info/operations_info.json", "w") as f:
    json.dump(ALL_FUNCS, f, indent=4)

# json_string = json.dumps(ALL_FUNCS, indent = 4)
# print(json_string)


# class CheckDecorators(ast.NodeVisitor):
#     def visit_FunctionDef(self, node):
#         if len(node.decorator_list) > 1:
#             print("function '%s' is decorated by: %s" % (node.name, ', '.join(ast.dump(decorator) for decorator in node.decorator_list if not isinstance(decorator, ast.Name) or decorator.id != 'check_decorators')))
#
# def check_decorators(func):
#     CheckDecorators().visit(ast.parse(dedent(inspect.getsource(func))))
#     return func
#
#
#
# def decorator(func):
#     def nested(*args, **kwargs):
#         return func(*args, **kwargs)
#     return nested
#
# @decorator
# @check_decorators
# def decorated():
#     pass

# dir_operations = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "src/datalab/operations/",
# )


# "edit" functions
# dir_operations_edit_general = dir_operations + "edit/general.py"
# sys.path.append("/usr1/data/pliu3/expressAI/Datalab/src/datalab/operations/edit")
# print(dir_operations_edit_general)
# file_name = my_module = importlib.import_module(dir_operations_edit_general)

# print(file_name)


# def import_from(module, name):
#     module = __import__(module, fromlist=[name])
#     return getattr(module, name)
#
# for file_name in os.listdir(general_dir):
#     if not file_name.endswith(".py") and file_name!="__pycache__":
#         print(f"{file_name}.transformation.{file_name}")
#         # my_module = importlib.import_module("abbreviate.transformation")
#         file_name = import_from(f"{file_name}.transformation", file_name)
#         print(file_name)
