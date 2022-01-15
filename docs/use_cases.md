# Use Case





### Get Dataset-level Statistics

#### Summarization
```python
from datalabs import load_dataset
from aggregate.summarization import *
dataset = load_dataset('xsum')
res = dataset['test'].apply(get_statistics)
```
Output:
```json
{'average_text_length': 361.06087877183694, 'average_summary_length': 21.100494088583023, 
 'density': 1.0623607565351458, 'coverage': 0.6602578726258126,
 'compression': 19.651115012606297, 'repetition': 9.453581574182701e-05, 'novelty': 0.8341764451842657, 'copy_len': 1.2924767321902775}
```