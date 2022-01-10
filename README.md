# DataLab API CN

## Installation
#### Install

    ```shell
    git clone https://github.com/ExpressAI/Datalab.git
    cd Datalab
    python setup.py install
    ```
   or 

    ```shell
    git clone https://github.com/ExpressAI/Datalab.git
    cd Datalab
    pip install .
    ```

#### Uninstall
    ```shell
    pip uninstall datasets
    ```

#### Dataset Operation
```python

# pip install datalab
from datalab import operations, load_dataset
from featurize import *

dataset = load_dataset("ag_news")
res = dataset["test"].apply(get_text_length)
print(next(res))


```

   

### Demo for add new datasets and write them into Mongodb

`python -m unittest tests/test_mongodb.py`，

```python
import datalab

dataset = datalab.load_dataset("datasets/adv_mtl")
dataset["imdb_test"].write_db()
print("OK")

# 这个函数用于删除上面产生的 dev_samples_of_dataset 的 adv_mtl 集合（不是清空），慎用
# cluster = datalab.MongoDBClient("cluster0")
# cluster.drop("dev_samples_of_dataset", "adv_mtl", True)

```


