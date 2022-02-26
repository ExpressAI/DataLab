# MongoDB Normalization: Feature Names


## 有哪些地方需要修改？


### dataset_metadata
`dataset_metadata` 里面保存了很多数据集的metadata信息，每个document对应一个数据集,由 `_id` 唯一确定
这个document里面, `features`保存了sample-level和dataset-level的特征，我们就是要归一化他们的names.



### samples_of_dataset
这里面每个collection对应的是一个数据集所有的样本，怎么知道一个样本对应的数据集是什么呢？可以通过 `dataset_id`去上面的 `dataset_metadata`里面找，是
一一对应的。那我们要改什么呢？就是每个collection (相当于一个数据集) 里面所有document （相当于一个样本）中的  `features`.

