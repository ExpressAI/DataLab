from __future__ import annotations

from collections.abc import Iterator
import os
import sys
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from operation import OperationFunction, TextOperation


class Data:
    def __init__(self, data: Iterator = None):

        self.data = data

    def apply(self, func: OperationFunction):
        # raise NotImplementedError
        raise NotImplementedError


class TextData(Data):
    def __init__(self, data: Iterator):
        # self.data = {"text":data}
        if isinstance(data, str):
            data = [data]

        self.data = [{"text": text} for text in data]

    def apply(self, func: TextOperation):

        if func._type == "Aggregating":
            yield func([text["text"] for text in self.data])
        elif func._type in [
            "Editing",
            "Preprocessing",
            "Featurizing",
            "OperationFunction",
        ]:
            for sample in self.data:
                yield func(sample[func.processed_fields[0]])
        else:
            for sample in self.data:
                yield func(sample)


class StructuredData(Data):
    def __init__(self, data: Iterator):
        # self.data = {"text":data}
        if isinstance(data, str):
            data = [data]

        self.data = [{"structured_data": structured_data} for structured_data in data]

    def apply(self, func: TextOperation):
        raise NotImplementedError


class StructuredTextData(StructuredData, TextData):
    def __init__(self, data: Iterator):

        super(StructuredTextData, self).__init__(data)

    def apply(self, func: OperationFunction):
        raise


"""
Data -> TextData -> Dataset -> TaskDataset
     -> Wikidata
     -> Image
     -> Video
"""
