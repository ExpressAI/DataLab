# How to Preprocess you Data?

 

```python
from datalabs import TextData
from featurize import *

a = TextData("I like this movie.")
b = a.apply(lower) # lowercase the text

print(next(b))

```