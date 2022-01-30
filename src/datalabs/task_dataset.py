

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datalabs import Dataset


class SequenceLabelingDataset(Dataset):
    def apply(self, func):
        if func._type == 'Aggregating':
            texts = [" ".join(tokens) for tokens in self["tokens"]] # [tokens] -> texts
            yield func(texts)
        elif func._type == "SequenceLabelingAggregating":
            yield func(self)
        elif func._type in ["Editing","Preprocessing", "Featurizing","OperationFunction"]:
            for sample in self.__iter__():
                yield func(" ".join(sample["tokens"])) # convert tokens -> a text
        else:
            for sample in self.__iter__():
                yield func(sample)


# class SummarizationDataset(Dataset):
#     def apply(self, func):
#         if func._type == 'Aggregating':
#             texts = [" ".join(tokens) for tokens in self["tokens"]] # [tokens] -> texts
#             yield func(texts)
#         elif func._type == "SummarizationAggregating":
#             yield func(self)
#         elif func._type in ["Editing","Preprocessing", "Featurizing","OperationFunction"]:
#             for sample in self.__iter__():
#                 yield func(" ".join(sample["tokens"])) # convert tokens -> a text
#         else:
#             for sample in self.__iter__():
#                 yield func(sample)