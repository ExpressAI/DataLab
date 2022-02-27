# MongoDB Normalization: Feature Names


## 有哪些地方需要修改？


### dataset_metadata
`dataset_metadata` 里面保存了很多数据集的metadata信息，每个document对应一个数据集,由 `_id` 唯一确定
这个document里面, `features`保存了sample-level和dataset-level的特征，我们就是要归一化他们的names.

比如，对于这个数据集`sst`, 我们先找到对应的描述其metadata信息的document,
![image](https://user-images.githubusercontent.com/59123869/155820388-f0c34705-dc05-411a-bbfa-c5e3dbd236c9.png)

然后看看这个数据集的features是否需要被归一化，比如这里的feature 用 `sentence` 来刻画文本信息，我们想把它变成`text`，我们就需要对以下包含 sentence的feature names进行统一替换：
![image](https://user-images.githubusercontent.com/59123869/155820453-f1e96071-a61e-440d-b599-28862d10b5df.png)





### samples_of_dataset
这里面每个collection对应的是一个数据集所有的样本，怎么知道一个样本对应的数据集是什么呢？可以通过 `dataset_id`去上面的 `dataset_metadata`里面找，是
一一对应的。那我们要改什么呢？就是每个collection (相当于一个数据集) 里面所有document （相当于一个样本）中的  `features`.

比如，对于上述数据集，我们改了`sst` metadata里面的feature names之后，我们还要把他具体样本里的feature name改下：步骤
（1）先在samples_of_dataset里面搜索名字为`sst`的collection，然后在这个`sst` collection里面过滤出来`dataset_id` = 上述metadata里面document `_id`的记录
（2）对上述过滤出来的记录的 `features`进行修改








## Other normalization
我们现在db 里的nli数据集都是这么分的。
task categories: text-classification
tasks : natural-language-inference