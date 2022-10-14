# 如何快速构建一个可交互的多数据集评估基准？([最终实现效果](https://explainaboard.inspiredco.ai/benchmark))

评估基准（Evaluation Benchmark）通常由多个典型数据集组成、用于
评估系统性能从而追踪某些领域的技术进展的基准。典型的例子如GLUE, XTREME.

本文档详细介绍了如何快速构建一个可交互的（e.g.,用户可以方便进行系统提交以及各种分析）、
可定制化（评估指标的选用、不同任务权重的设计）的多数据集评估基准。

### 1. 添加数据集到 DataLab SDK

第一步是按照[doc](https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md)
将基准测试中涉及的所有数据集添加到DataLab SDK中

### 2. 任务评估功能实现

检查现有的[ExplainaBoard SDK](https://github.com/neulab/ExplainaBoard/blob/7fb8ccd2b999f5eb831ebae6011cee2dfff393fe/explainaboard/constants.py#L4)是否支持不同数据集的任务

* 如果所有都支持，则跳转到下一步
* 如果有任何任务不被支持，那么你需要通过这个[doc](https://github.com/neulab/ExplainaBoard/blob/main/docs/add_new_tasks.md)添加一个新的任务到ExplainaBoard SDK中
。如果您在此过程中有任何困难，请随时与我们联系(例如，新建一个`issue`)。

### 3. 创建一个基准配置文件 （benchmark config）并发出一个`pull request`

将基准测试的所有数据集添加到DataLab SDK中，并且ExplainaBoard SDK支持所有任务之后，
下一步是创建一个基准配置文件：

* 基准是如何由这些数据集组成的
* 每个任务将使用什么评估指标

我们将使用一个示例来展示配置文件的样子：

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

* `id`: 基准的id.
* `name`: 基准的名字
* `visibility`: 这可能是“private”或“public”，表明这个基准是否可以在网络平台上被观察到。
* `contact`: 基准的联系人信息（一般写邮箱即可）
* `homepage`: 基准的主页信息
* `datasets`: 它是一个列表，其中每个元素记录数据集的元数据信息。
  * `dataset_name`: dataset name
  * `sub_dataset`: sub_dataset name
  * `dataset_split`: train|validation|test
  * `metrics`: it is a list, where each element specifies a metric name
  * `task`: task name
  * `output_file_type`: the format of output file
* `views`: 这指定了基准测试在网页中显示时的布局和聚合策略。它可以是一个列表，这意味着支持多种聚合方式。

更多基准配置文件的[例子](https://github.com/neulab/explainaboard_web/tree/main/backend/src/impl/benchmark_configs)

一旦您成功地创建了基准配置，可以在[ExplainaBoard Web](https://github.com/neulab/explainaboard_web/pulls)上发起一个pull request，并放置 [这里](https://github.com/neulab/explainaboard_web/tree/main/backend/src/impl/benchmark_configs).
