from typing import Dict, List, Any, Optional
from typing import Callable, Mapping
from datalabs.operations.featurize.text_classification import TextClassificationFeaturizing, text_classification_featurizing


class WikipediaFeaturizing(TextClassificationFeaturizing):


    def __init__(self, *args, **kwargs
                 ):

        super(AGNewsFeaturizing, self).__init__(*args, **kwargs)
        self._type = 'AGNewsFeaturizing'
        self._data_type = "ag_news"



class wikipedia_featurizing(text_classification_featurizing):
    def __init__(self, *args, **kwargs
                 ):


        super(ag_news_featurizing, self).__init__(*args, **kwargs)
        # print(self.__dict__)


    def __call__(self, *param_arg):
        if callable(self.name):

            tf_class = WikipediaFeaturizing(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]

            name = self.name or f.__name__
            tf_cls = WikipediaFeaturizing(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                                    processed_fields = self.processed_fields,
                                    generated_field = self.generated_field,
                                    task = self.task,
                                    description=self.description,
                                    )
            return tf_cls



"""
from datalabs import load_dataset
dataset = load_dataset('./ag_news')
from ag_news.featurize import *
get_number_of_tokens.__dict__
res = dataset['test'].apply(get_number_of_tokens)
print(next(res))
"""



@wikipedia_featurizing(name = "get_number_of_tokens", contributor= "datalab", processed_fields= "text",
                                 task="text-classification", description="this function is used to calculate the text length",
                                 )
def get_number_of_tokens(sample:dict):
    return len(sample['text'])


