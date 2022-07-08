"""XWikis: A Multilingual Abstractive Summarization Dataset"""
import os
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{perez-beltrachini-lapata-2021-models,
    title = "Models and Datasets for Cross-Lingual Summarisation",
    author = "Perez-Beltrachini, Laura  and
      Lapata, Mirella",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.742",
    doi = "10.18653/v1/2021.emnlp-main.742",
    pages = "9408--9423",
    abstract = "We present a cross-lingual summarisation corpus with long documents in a source language associated with multi-sentence summaries in a target language. The corpus covers twelve language pairs and directions for four European languages, namely Czech, English, French and German, and the methodology for its creation can be applied to several other languages. We derive cross-lingual document-summary instances from Wikipedia by combining lead paragraphs and articles{'} bodies from language aligned Wikipedia titles. We analyse the proposed cross-lingual summarisation task with automatic metrics and validate it with a human study. To illustrate the utility of our dataset we report experiments with multi-lingual pre-trained models in supervised, zero- and few-shot, and out-of-domain scenarios.",
}
"""

_DESCRIPTION = """\
XWikis is a cross-lingual summarisation corpus with long documents in a source language associated with multi-sentence summaries in a target language. 
The corpus covers twelve language pairs and directions for four European languages, namely Czech, English, French and German, and the methodology for its creation can be applied to several other languages. 
Cross-lingual document-summary instances are derived from Wikipedia by combining lead paragraphs and articles bodies from language aligned Wikipedia titles.
see: https://aclanthology.org/2021.emnlp-main.742.pdf
"""

_HOMEPAGE = "https://github.com/lauhaide/clads"
_LICENSE = "MIT License"
_ABSTRACT = "summary"
_ARTICLE = "text"



class XWikisConfig(datalabs.BuilderConfig):
    """BuilderConfig for XWikis."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for XWikis.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(XWikisConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class XWikisDataset(datalabs.GeneratorBasedBuilder):
    """XWikis Dataset."""
    # _LANG = ["fr", "de", "cs"]
    _LANG = ["cs"]
    _URLs = {
        # "fr-en": "https://datashare.ed.ac.uk/bitstream/handle/10283/4188/XWikis-prepa-fr-en.zip?sequence=21&isAllowed=y",
        # "de-en": "https://datashare.ed.ac.uk/bitstream/handle/10283/4188/XWikis-prepa-de-en.zip?sequence=20&isAllowed=y",
        "cs-en": "https://datashare.ed.ac.uk/bitstream/handle/10283/4188/XWikis-prepa-cs-en.zip?sequence=19&isAllowed=y",
    }
    BUILDER_CONFIGS = list([
        XWikisConfig(
            name=f"{l}-en",
            version=datalabs.Version("1.0.0"),
            description=f"XWikis Dataset for crosslingual summarization, {l}-en split",
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT),
            ]
        ) for l in _LANG
    ])
    DEFAULT_CONFIG_NAME = "cs-en"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            license=_LICENSE,
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)],
            languages=self.config.name.split('-'),

        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(self._URLs[self.config.name])
        # cross-lingual summarization
        src_id, tgt_id = self.config.name.split("-")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "text_path": os.path.join(f_path, f"./XWikis-prepa/{src_id}-{tgt_id}/train.{src_id}_{tgt_id}_src_documents.txt"), 
                    "summary_path": os.path.join(f_path, f"./XWikis-prepa/{src_id}-{tgt_id}/train.{src_id}_{tgt_id}_tgt_summaries.txt")
                    }
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "text_path": os.path.join(f_path, f"./XWikis-prepa/{src_id}-{tgt_id}/val.{src_id}_{tgt_id}_src_documents.txt"),
                    "summary_path": os.path.join(f_path, f"./XWikis-prepa/{src_id}-{tgt_id}/val.{src_id}_{tgt_id}_tgt_summaries.txt")
                    }
            ),
            # datalabs.SplitGenerator(
            #     name=datalabs.Split.TEST,
            #     gen_kwargs={
            #         "text_path": os.path.join(f_path, f"./XWikis-prepa/{src_id}-{tgt_id}/test.{src_id}_{tgt_id}_src_documents.txt"),
            #         "summary_path": os.path.join(f_path, f"./XWikis-prepa/{src_id}-{tgt_id}/test.{src_id}_{tgt_id}_tgt_summaries.txt")
            #         }
            # ),
        ]

    def _generate_examples(self, text_path, summary_path):
        """Generate XWikis examples."""
        with open(text_path, encoding="utf-8") as f_src, open(summary_path, encoding="utf-8") as f_tgt: 
            for (id_, (x, y)) in enumerate(zip(f_src, f_tgt)):
                x = x.strip()
                y = y.strip()
                yield id_, {_ARTICLE: x, _ABSTRACT: y}
                