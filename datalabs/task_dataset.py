from datalabs import Dataset


class SequenceLabelingDataset(Dataset):
    def func1(self):
        return 0

    # def apply(self, func):
    #     if func._type == 'Aggregating':
    #         texts = [" ".join(tokens) for tokens in self["tokens"]] #
    #         [tokens] -> texts
    #         yield func(texts)
    #     elif func._type == "SequenceLabelingAggregating":
    #         yield func(self)
    #     elif func._type in ["Editing","Preprocessing", "Featurizing",
    #     "OperationFunction"]:
    #         for sample in self.__iter__():
    #             yield func(" ".join(sample["tokens"])) # convert tokens -> a text
    #     else:
    #         for sample in self.__iter__():
    #             yield func(sample)


class TextMatchingDataset(Dataset):
    def func1(self):
        return 0

    # def apply(self, func):
    #     if func._type == 'Aggregating':
    #         texts = [text + " " + text2 for text1, text2 in zip(self["text1"],
    #         self['text2'])] # [tokens] -> texts
    #         yield func(texts)
    #     elif func._type == "TextMatchingAggregating":
    #         yield func(self)
    #     elif func._type in ["Editing","Preprocessing", "Featurizing",
    #     "OperationFunction"]:
    #         for sample in self.__iter__():
    #             if func.processed_fields[0] =="text":
    #                 yield func(sample["text1"]) # convert tokens -> a text
    #             else:
    #                 yield func(sample[func.processed_fields[0]])
    # convert tokens -> a text
    #     else:
    #         for sample in self.__iter__():
    #             yield func(sample)


# class SummarizationDataset(Dataset):
#     def apply(self, func):
#         if func._type == 'Aggregating':
#             texts = [" ".join(tokens) for tokens in self["tokens"]]
# [tokens] -> texts
#             yield func(texts)
#         elif func._type == "SummarizationAggregating":
#             yield func(self)
#         elif func._type in ["Editing","Preprocessing", "Featurizing",
#         "OperationFunction"]:
#             for sample in self.__iter__():
#                 yield func(" ".join(sample["tokens"])) # convert tokens -> a text
#         else:
#             for sample in self.__iter__():
#                 yield func(sample)
