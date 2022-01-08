from typing import Any, ClassVar, Dict, List, Optional
from typing import Callable, Mapping, Iterator
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from operation  import OperationFunction, TextOperation


class Data:
    def __init__(self, data:Iterator = None, name:str = "Data"):
        self.name = name
        self.data = data

    def apply(self, func:OperationFunction):
        #raise NotImplementedError
        for sample in self.data:
            #yield func(sample)
            if len(func.processed_fields)==1 and sample.keys():
                yield func(sample[func.processed_fields[0]])
            else:
                yield func(sample)



class TextData(Data):
    def __init__(self, data:Iterator, name:str = "textData"):
        # self.data = {"text":data}
        if isinstance(data, str):
            data = [data]

        self.data = [{"text":text} for text in data]

        self.name = name


    def apply(self, func: TextOperation):
        for sample in self.data:
            #yield func(sample)
            if len(func.processed_fields)==1 and sample.keys():
                yield func(sample[func.processed_fields[0]])
            else:
                yield func(sample)



class StructuredData(Data):
    def __init__(self, data:Iterator, name:str = "StructuredData"):
        # self.data = {"text":data}
        if isinstance(data, str):
            data = [data]

        self.data = [{"structured_data":structured_data} for structured_data in data]

        self.name = name


    def apply(self, func: StructuredData):
        for sample in self.data:
            #yield func(sample)
            if len(func.processed_fields)==1 and sample.keys():
                yield func(sample[func.processed_fields[0]])
            else:
                yield func(sample)



class XMLData(TextData):
    def __init__(self, data:Iterator, name:str = "XMLData"):
        # self.data = {"text":data}
        if isinstance(data, str):
            data = [data]

        self.data = [{"text":text} for text in data]

        self.name = name


    def apply(self, func: OperationFunction):
        for sample in self.data:
            #yield func(sample)
            if len(func.processed_fields)==1 and sample.keys():
                yield func(sample[func.processed_fields[0]])
            else:
                yield func(sample)



"""
Data -> TextData -> Dataset -> TaskDataset
     -> Wikidata
     -> Image
     -> Video
"""
