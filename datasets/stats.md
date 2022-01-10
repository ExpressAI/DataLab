
```python

"""
from datalab import load_dataset
dataset = load_dataset("./app_reviews")
dataset["test"]._info.task_templates

dataset['train']._info.task_templates

"""

```


#### ag_news
label = ["World", "Sports", "Business", "Science and Technology"]


#### adv_mtl
    textualize_label = {"1":"positive",
                         "0":"negative"}


##### app_reviews
        textualize_label = {"1":"1 star",
                                 "2":"2 stars",
                                 "3":"3 stars",
                                 "4":"4 stars",
                                 "5":"5 stars"}



#### tweet_eval
dataset = load_dataset("./tweet_eval","emoji")


#### amazon_reviews_multi
* a mixture of multiple languages


#### banking77
* category names contain underscore, for example `transfer_timing`

#### yahoo_answers_topics
* category names contain `&`, for example: 'Computers & Internet'


#### financial_phrasebank
* subset = ['sentences_allagree', 'sentences_75agree', 'sentences_66agree', 'sentences_50agree']



#### GLUE
* stsb: label_names = None