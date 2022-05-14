import json
import datalabs
from datalabs import get_task, TaskType

from datalabs.utils.logging import get_logger

logger = get_logger(__name__)


_CITATION = """
@article{wang2022mconala,
  title={MCoNaLa: A Benchmark for Code Generation from Multiple Natural Languages},
  author={Zhiruo Wang and Grace Cuenca and Shuyan Zhou and Frank F. Xu and Graham Neubig},
  journal={arXiv preprint arXiv:2203.08388},
  year={2022}
}
"""


_DESCRIPTION = """\
MCoNaLa was designed to test systems for generating program snippets from multiple 
natural languages besides English (Spanish, Japanese, and Russian). For example, 
if the input is sort list x in reverse order, then the system would be required to 
output x.sort(reverse=True) in Python. It includes three datasets crawled from 
Stack Overflow in respective languages, automatically filtered, then curated by 
annotators, yielding 341 Spanish, 210 Japanese, and 345 Russian samples for testing. 
"""
 

LANG_URLS = {
  "es": "https://raw.githubusercontent.com/zorazrw/multilingual-conala/master/dataset/test/es_test.json", 
  "ja": "https://raw.githubusercontent.com/zorazrw/multilingual-conala/master/dataset/test/ja_test.json", 
  "ru": "https://raw.githubusercontent.com/zorazrw/multilingual-conala/master/dataset/test/ru_test.json", 
}


class MConalaConfig(datalabs.BuilderConfig):
    """BuilderConfig for MConala."""

    def __init__(self, **kwargs):
        """BuilderConfig for Conala.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MConalaConfig, self).__init__(**kwargs)



class MConala(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name=lang,
        )
        for lang in list(LANG_URLS.keys())
    ]

    def _info(self):
        features_dataset = datalabs.Features()
        features_sample = datalabs.Features(
            {
                "question_id": datalabs.Value("int32"),
                f"orig_{self.config.name}": datalabs.Value("string"),
                "translation": {
                    self.config.name: datalabs.Value("string"),
                    "python": datalabs.Value("string"),
                }
            }
        )

        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            citation=_CITATION,
            # Homepage of the dataset for documentation
            homepage="https://github.com/zorazrw/multilingual-conala/",
            # datasets.features.FeatureConnectors
            features=features_sample,
            features_dataset=features_dataset,
            supervised_keys=None,
            languages=[self.config.name,'python'],
            task_templates=[
                get_task(TaskType.code_generation)(
                    translation_column="translation",
                    lang_sub_columns=[self.config.name,'python'],
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        lang = self.config.name 
        dataset_path = dl_manager.download_and_extract(LANG_URLS[lang])

        return [
            datalabs.SplitGenerator(
                name='test',
                gen_kwargs={"filepath": f'{dataset_path}'},
            ), 
        ]

    def _convert_data(self, data):
        return {
            "question_id": data['question_id'],
            f"orig_{self.config.name}": data['intent'],
            "translation": {
                self.config.name: data['rewritten_intent'],
                "python": data['snippet'],
            }
        }

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, 'r') as fin: 
            for _id, data in enumerate(json.load(fin)): 
                yield _id, self._convert_data(data)