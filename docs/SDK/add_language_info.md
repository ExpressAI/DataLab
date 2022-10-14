# How to add language information of you dataset?

Take the `ag_news` (English), for example,

we just need to assign value (`List[str]`) to the variable `languages` in the `_info(self)` function.

* where `str` represents the ISO code of a language, which you can obtain from this [look-up table](https://huggingface.co/languages).
If the dataset involves multiple languages, add all of them. For example, for the Chinese-English machine translation dataset, we have
  `languages = ['en','zh']`

```python
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["World", "Sports", "Business", "Science and Technology"]),
                }
            ),
            homepage="http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html",
            citation=_CITATION,
            languages=["en"],
            task_templates=[TextClassification(text_column="text", label_column="label")],
        )

```

You can verify if the language information has been successfully added by:

```python
from datalabs import load_dataset

dataset = load_dataset("ag_news")
print(dataset['test']._info.languages)
```

where `dataset['test']._info` represents all metadata information of the test split.
