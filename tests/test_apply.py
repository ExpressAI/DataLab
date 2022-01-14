"""
from datalabs import load_dataset, TextData
from aggregate import *
dataset = load_dataset('mr')
res = dataset['test'].apply(get_average_length)




from datalabs import TextData
from aggregate import *
a = TextData(["I think this is a good movie", "this is not a good movie"])
res = a.apply(get_tfidf)
print(next(res))

"""