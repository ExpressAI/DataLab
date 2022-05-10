# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Dict, List, Any, Optional
from typing import Callable, Mapping, Iterator
from tqdm import tqdm
from datalabs.operations.aggregate import Aggregating, aggregating


class QAMultipleChoiceAggregating(Aggregating):


    def __init__(self, *args, **kwargs
                 ):

        super(QAMultipleChoiceAggregating, self).__init__(*args, **kwargs)
        self._data_type = "dataset"



class qa_multiple_choice_aggregating(aggregating):
    def __init__(self, *args, **kwargs
                 ):


        super(qa_multiple_choice_aggregating, self).__init__(*args, **kwargs)
        # print(self.__dict__)


    def __call__(self, *param_arg):
        if callable(self.name):

            tf_class = QAMultipleChoiceAggregating(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = QAMultipleChoiceAggregating(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                                   task = self.task,
                                   description= self.description,)
            return tf_cls



