import os

"""

data = load_data("wikipedia")


"""



"""
if the task's column haven't been specified, the function will be 
applied based on the feature type of text, for example, string will be
"""
@datalab.preprocessing
def tokenize(text):
    return spacy.tokenize(text)

@datalab.augmentation
def token_replace(text):
    return checklist.replace(text)

@datalab.preprocessing(TASK.SUMMARIZATION.text_column)
def tokenize(document):
    return spacy.tokenize(document)

@datalab.preprocessing(TASK.SUMMARIZATION.summary_column)
def tokenize(document):
    return spacy.tokenize(document)


@datalab.prompting(TASK.NLI)
def template1(text1, text2, labels):
    return text1 + text2 + " ".join(labels)

@datalabs.feature
def get_length(text):
    return len(text.split(" "))


dataset = load_data("adv_mtl")
new_dataset = dataset["imdb_test"].apply(template1)




"""
import numpy as np
a = [[1,2,3], [4,5,6]]
A = np.array(a)
A.sum()
"""

"""
import datalab as dl
b = ["This is the first sentence", "This is the second sentence"]
B = dl.data(b) # Data
B.apply(preprocessing.tokenize)
B.apply(feature.ner)
"""

"""
    from datalab import load_dataset
    from datalab.operations.featurize.summarization import get_compression

    dataset = load_dataset("cnn_dailymail", "3.0.0")
    res_iterator = dataset["test"].apply(get_compression)


"""







