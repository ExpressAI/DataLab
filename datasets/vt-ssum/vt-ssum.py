"""VT-SSum: A Benchmark Dataset for Video Transcript Segmentation and Summarization"""
import os
import glob
import json
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@article{lv2021vt,
  title={VT-SSum: A Benchmark Dataset for Video Transcript Segmentation and Summarization},
  author={Lv, Tengchao and Cui, Lei and Vasilijevic, Momcilo and Wei, Furu},
  journal={arXiv preprint arXiv:2106.05606},
  year={2021}
}
"""

_DESCRIPTION = """\
We present VT-SSum, a benchmark dataset with spoken language for video transcript segmentation and summarization,
which includes 125K transcript-summary pairs from 9,616 videos.
see: https://arxiv.org/pdf/2106.05606.pdf
"""

_HOMEPAGE = "https://github.com/Dod-o/VT-SSum"
_LICENSE = "CC BY-NC-ND 4.0"
_ARTICLE = "text"
_ABSTRACT = "summary"


class VTSSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for VTSSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for VTSSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(VTSSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class VTSSumDataset(datalabs.GeneratorBasedBuilder):
    """VTSSum Dataset."""

    BUILDER_CONFIGS = [
        VTSSumConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="Video transcript extractive summarization dataset.",
            task_templates=[get_task(TaskType.extractive_summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        )
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Sequence(datalabs.Value("string"))
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            license=_LICENSE,
            languages=["en"],
            task_templates=self.config.task_templates
        )

    def _split_generators(self, dl_manager):
        url = "https://github.com/Dod-o/VT-SSum/archive/refs/heads/main.zip"
        f_path = dl_manager.download_and_extract(url)

        train_f_path = os.path.join(f_path, "VT-SSum-main/train")
        valid_f_path = os.path.join(f_path, "VT-SSum-main/dev")
        test_f_path = os.path.join(f_path, "VT-SSum-main/test")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": train_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"f_path": valid_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": test_f_path}
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate VT-SSum examples."""

        files = glob.glob(os.path.join(f_path, "*.json"))
        datas = []

        for file in files:
            text = []
            summary = []
            f = open(file, encoding="utf-8")
            clips = json.load(f)["summarization"]

            for index in range(len(clips)):
                clip = clips["clip_{}".format(index)]
                summarization_data = clip["summarization_data"]
                for one in summarization_data:
                    text.append(one["sent"])
                    if one["label"] == 1:
                        summary.append(one["sent"])
            text = " ".join(text)
            datas.append((text, summary))

        for id_, (text, summary) in enumerate(datas):
            raw_feature_info = {
                _ARTICLE: text,
                _ABSTRACT: summary
            }
            yield id_, raw_feature_info
