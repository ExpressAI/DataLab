# How to use DataLab to Process your Data?


Another key feature of DataLab is the standardization of different data operations into a unified 
format to satisfy different data processing requirements in one place.
To this end, we devised a general typology for the concepts of data and operation as shown below and curated schemas for these objects.


<img src="https://user-images.githubusercontent.com/59123869/155357470-8b95671c-d5e4-45bb-9edf-076d33c1e6f2.png" width="600">


Specifically, using [DataLab SDK](https://github.com/ExpressAI/DataLab), you can conveniently 
* process data with a unified interface
* and use rich operations supported by SDK by passing different operation names


<img src="https://user-images.githubusercontent.com/59123869/155447749-457820f2-d0e5-4426-acb2-17ea3bdff011.png" width="600">

 

You can install the SDK by:
```
pip install --upgrade pip
pip install datalabs
```



So far, DataLab supports following types of data operations and implements some functions for each operation. Users can continue expanding 
them by yourselves.


### Preprocessing

Data preprocessing (e.g., tokenization) is an indispensable step in training deep  learning and machine learning models, 
and the quality of the dataset directly affects the learning of models. Currently, DATALAB supports both general preprocessing functions
and task-specific ones, which are built based on different sources, such as Spacy, NLTK and Huggingface.


<img src="https://user-images.githubusercontent.com/59123869/155447787-063de6fa-f6aa-4377-88d2-f985d5ff080c.png" width="600">

### Editing (Transformation)
Editing aims to apply certain transformations to a given text, which spans multiple important applications in NLP, 
for example 
* adversarial evaluation [(Ribeiro et al., 2021)](https://arxiv.org/pdf/2005.04118.pdf), which usually requires diverse perturbations on test samples to test the robustness of a system.
* data augmentation [(Dhole et al., 2021)](https://arxiv.org/pdf/2112.02721.pdf). 
Essentially, many of the methods for constructing augmented or diagnostic datasets involve some editing operation on the original dataset
  (e.g., named entity replacement in diagnostic dataset construction [(Ribeiro et al., 2021)](https://arxiv.org/pdf/2005.04118.pdf), token deletion in data augmentation [jason_2019](https://aclanthology.org/D19-1670.pdf).
DataLab provides a unified interface for data editing and users can easily apply to edit the data they are interested in.
  
 


<img src="https://user-images.githubusercontent.com/59123869/155447848-794b6f94-e280-4874-953c-b7cfd02b2d65.png" width="600">


  
### Featurizing
This operation aims to compute sample-level features of a given text.
In DataLab, in addition to designing some general feature functions (e.g. `*get_length()*` operation 
calculates the length of the text.), we also customize some feature functions for specific tasks (e.g. `*get_oracle()*`
operation for the summarization task that calculates the oracle summary of the source text.).

 


<img src="https://user-images.githubusercontent.com/59123869/155447958-74deb639-8ff1-49cf-88fa-fa82439cc68c.png" width="600">


### Aggregating
Aggregation operations are used to compute corpus-level statistics such as TF-IDF, 
label distribution. Currently, DataLab supports both generic aggregation operations applicable to any task and some customized ones
for four NLP tasks (classification, summarization, extractive question answering and natural language inference).




### Prompting

Prompt-based learning [Liu et al. 2021](https://arxiv.org/pdf/2107.13586.pdf) has received considerable attention, as better utilization of pretrained 
language models benefits many NLP tasks.
So far, DataLab includes different prompt templates  which can be applied to five types of tasks 
(topic classification, sentiment classification, sentence entailment, 
summarization, natural language inference).




