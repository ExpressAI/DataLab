"""

## Test for Prompt


from datalabs import load_dataset
dataset = load_dataset('xsum')
from prompt.summarization import *
res = dataset['test'].apply(template_p1)
print(next(res))



from datalabs import load_dataset
dataset = load_dataset('xsum')
from prompt.summarization import *
res = dataset['test'].apply(template_p1)
print(next(res))



from datalabs import load_dataset
dataset = load_dataset('sick')
from prompt.natural_language_inference import *
res = dataset['test'].apply(template_p1)
print(next(res))




from datalabs import load_dataset
dataset = load_dataset('mr')
from prompt.sentiment_classification import *
res = dataset['test'].apply(template_p1)
print(next(res))



from datalabs import load_dataset
from aggregate.text_classification import *
dataset = load_dataset('mr')
res = dataset['test'].apply(get_statistics)





from datalabs import load_dataset
from aggregate.summarization import *
dataset = load_dataset('xsum')
res = dataset['test'].apply(get_statistics)

from datalabs import load_dataset
from aggregate.text_matching import *
dataset = load_dataset('glue', 'mrpc')
res = dataset['test'].apply(get_statistics)
print(next(res))


from datalabs import load_dataset
from aggregate.sequence_labeling import *
dataset = load_dataset('wnut_17')
res = dataset['test'].apply(get_statistics)
print(next(res))


from datalabs import TextData
from aggregate import *
a = TextData(["I think this is a good movie", "this is not a good movie"])
res = a.apply(get_tfidf)
print(next(res))


as_dataset ->  _build_single_dataset -> _as_dataset ->  Dataset


"""