# How to create an evaluation benchmark?

An evaluation benchmark is usually composed of multiple typical datasets and used
to assess system performance, therefore tracking the technical progress in some areas.

This document details how to build an interactive benchmark for your evaluation purpose.

### 1. Add your datasets to DataLab SDK
The first step is to add all datasets involved in your benchmark into DataLab SDK by
following the [doc](https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md)


### 2. Task implementation
Check if the tasks for different datasets are supported by existing [ExplainaBoard SDK](https://github.com/neulab/ExplainaBoard/blob/7fb8ccd2b999f5eb831ebae6011cee2dfff393fe/explainaboard/constants.py#L4), 
* if all are supported, jump to the next step
* if there is any task that hasn't been supported, then you need to add a new task into
ExplainaBoard SDK by following this [doc](https://github.com/neulab/ExplainaBoard/blob/main/docs/add_new_tasks.md). If 
you have any trouble in doing this, feel free to contact (e.g., open an issue) us.

### 3. Create a benchmark config file and make a pull request
Once all datasets of the benchmark are added into DataLab SDK, and all tasks are supported by ExplainaBoard SDK,
the next step is to create a config that defines
 * how the benchmark is composed of these datasets
 * what evaluation metrics would be used for each task
 
We will use an example to show what the config file looks like.


```JSON
{
  "id": "gaokao",
  "name": "Gaokao",
  "visibility": "public",
  "contact":"stefanpengfei@gmail.com",
  "homepage":"https://github.com/gaokao",
  "paper":{
    "title":"Gaokao Benchmark",
    "url":"www.xxx.com"
  },
  "description":"Gaokao is a benchmark that can track how well we make progress towards human-level AI.",
  "logo":"https://explainaboard.s3.amazonaws.com/benchmarks/figures/gaokao2022.png",
  "datasets": [
    {
      "dataset_name": "gaokao2018_np1",
      "sub_dataset": "listening",
      "dataset_split": "test",
      "metrics": [{"name":"CorrectCount"}],
      "task": "qa-multiple-choice",
      "output_file_type": "json"
    },
    {
      "dataset_name": "gaokao2018_np1",
      "sub_dataset": "cloze-multiple-choice",
      "dataset_split": "test",
      "metrics": [{"name":"CorrectCount"}],
      "task": "cloze-multiple-choice",
      "output_file_type": "json"
    }
  ],
"views": [
      {
          "name": "Overall",
          "operations": [
              {"op": "weighted_sum", "group_by": ["dataset_name"], "weight": "sub_dataset", "weight_map": {
                  "listening":  1.5,
                   "cloze-multiple-choice": 1.5,
                   "cloze-hint":1.5,
                   "reading-multiple-choice":2,
                   "reading-cloze":2,
                   "writing-grammar":1,
                   "writing-essay":1
                   }},
                {"op": "mean"}
          ]
      }
]
}
``` 

where 
* `id`: the identify of your benchmark.
* `name`: benchmark name
* `visibility`: this could be "public" or "private", suggesting if this benchmark could be observed in
the web platform.
* `contact`: benchmark contact
* `homepage`: homepage information of the benchmark
* `datasets`: it's a list, where each element record the metadata information of a dataset.
    * `dataset_name`: dataset name
    * `sub_dataset`: sub_dataset name
    * `dataset_split`: train|validation|test
    * `metrics`: it is a list, where each element specifies a metric name
    * `task`: task name
    * `output_file_type`: the format of output file
* `views`: this specifies the layout and aggregation strategies of benchmark when displayed in the web page.
It could be a list, meaning that multiple aggregation ways are supported.

 
 More examples could be found [here](https://github.com/neulab/explainaboard_web/tree/main/backend/src/impl/benchmark_configs)
 
 
Once you have successfully created the benchmark config, you could set up a pull request at [ExplainaBoard Web](https://github.com/neulab/explainaboard_web/pulls)
and put your config [here](https://github.com/neulab/explainaboard_web/tree/main/backend/src/impl/benchmark_configs).