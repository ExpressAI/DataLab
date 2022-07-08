import json
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """
@inproceedings {xu-etal-2020-clue,
 title = "{CLUE}: A {C}hinese Language Understanding Evaluation Benchmark",
 author = "Xu, Liang  and
    Hu, Hai and
    Zhang, Xuanwei and
    Li, Lu and
    Cao, Chenjie and
    Li, Yudong and
    Xu, Yechen and
    Sun, Kai and
    Yu, Dian and
    Yu, Cong and
    Tian, Yin and
    Dong, Qianqian and
    Liu, Weitang and
    Shi, Bo and
    Cui, Yiming and
    Li, Junyi and
    Zeng, Jun and
    Wang, Rongzhao and
    Xie, Weijian and
    Li, Yanting and
    Patterson, Yina and
    Tian, Zuoyu and
    Zhang, Yiwen and
    Zhou, He and
    Liu, Shaoweihua and
    Zhao, Zhe and
    Zhao, Qipeng and
    Yue, Cong and
    Zhang, Xinrui and
    Yang, Zhengliang and
    Richardson, Kyle and
    Lan, Zhenzhong ",
 booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
 month = dec,
 year = "2020",
 address = "Barcelona, Spain (Online)",
 publisher = "International Committee on Computational Linguistics",
 url = "https://aclanthology.org/2020.coling-main.419",
 doi = "10.18653/v1/2020.coling-main.419",
 pages = "4762--4772",
 abstract = "The advent of natural language understanding (NLU) benchmarks for English, such as GLUE and SuperGLUE allows new NLU models to be evaluated across a diverse set of tasks. These comprehensive benchmarks have facilitated a broad range of research and applications in natural language processing (NLP). The problem, however, is that most such benchmarks are limited to English, which has made it difficult to replicate many of the successes in English NLU for other languages. To help remedy this issue, we introduce the first large-scale Chinese Language Understanding Evaluation (CLUE) benchmark. CLUE is an open-ended, community-driven project that brings together 9 tasks spanning several well-established single-sentence/sentence-pair classification tasks, as well as machine reading comprehension, all on original Chinese text. To establish results on these tasks, we report scores using an exhaustive set of current state-of-the-art pre-trained Chinese models (9 in total). We also introduce a number of supplementary datasets and additional tools to help facilitate further progress on Chinese NLU. Our benchmark is released at https://www.cluebenchmarks.com",
}"""

_DESCRIPTION = """
CLUE组织提供的蚂蚁金融语义相似度数据集(Ant Financial Question Matching Corpus)"""


class AFQMC(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text1": datalabs.Value("string"),
                    "text2": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["0", "1"]),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/CLUEbenchmark/CLUE",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.natural_language_inference)(
                    text1_column="text1", text2_column="text2", label_column="label"
                ),
            ],
        )

    def _split_generators(self, dl_manager):
        files_to_download = {
            "test": "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/afqmc/test.json",
            "dev": "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/afqmc/dev.json",
            "train": "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/afqmc/train.json",
        }
        downloaded_files = dl_manager.download_and_extract(files_to_download)
        # train_path = os.path.join(path, 'train.json')
        # dev_path = os.path.join(path, 'dev.json')
        # test_path = './test.json'
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                text1, text2, label = (
                    line["sentence1"],
                    line["sentence2"],
                    line["label"],
                )
                label = int(label)
                yield id_, {"text1": text1, "text2": text2, "label": label}
