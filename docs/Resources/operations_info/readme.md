# Information for Operations

suppose that

```python
 metadata =    {
        "class_type": "aggregating",
        "args": {
            "name": "get_statistics",
            "contributor": "datalab",
            "task": "text-matching, natural-language-inference",
            "description": "Calculate the overall statistics (e.g., average length) of a given text pair classification datasets. e,g. natural language inference"
        }
    }
```

we should have

```python
# pip install datalabs
from datalabs import load_dataset
from aggregate.text_matching import *

dataset = load_dataset("ag_news")
res=dataset["train"].apply(get_statistics)
```

Code:

```python
cmd_ops = ""
if metadata["class_type"] == "aggregating":
    pkg = metadata["args"]["task"].split(",")[0].replace("-", "_")
    cmd_ops = "from aggregate." + pkg + " import *"
```
