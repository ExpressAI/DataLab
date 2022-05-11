import json

import requests

# Search by language
path = "https://datalab.nlpedia.ai/api/fetch_data_for_sdk_language"
data = {"language": "fr"}
r = requests.post(path, json=data)
list_ = json.loads(r.content)
print(len(list_))
print(list_[0])


# Search by task
path = "https://datalab.nlpedia.ai/api/fetch_data_for_sdk_task"
data = {
    "task": "sentiment-classification",
}
r = requests.post(path, json=data)
list_ = json.loads(r.content)
print(len(list_))
print(list_[0])


# Get all languages, tasks and task_category
path = "https://datalab.nlpedia.ai/api/fetch_meta_data_for_sdk_task_language"
r = requests.get(path)
dic_ = json.loads(r.content)
print(dic_["languages"])  # list of all languages
print(dic_["tasks"])  # list of all tasks
print(dic_["task_categories"])  # list of all task_category
