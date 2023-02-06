"""WMT20 neural machine translation metrics dataset."""
import csv
import os

import datalabs
from datalabs import get_task, TaskType

logger = datalabs.logging.get_logger(__name__)

_DL_URLS = {
    "scores": "https://raw.githubusercontent.com/WMT-Metrics-task/wmt20-metrics/main/manual-evaluation/DA/",  # noqa
    "text": "https://drive.google.com/uc?export=download&confirm=t&id=1P-Y1P-GTMCNtWj8qaeq-U-m-0DGGnOaP",  # noqa
}

_DESCRIPTION = """\
This shared task will examine automatic evaluation metrics
for machine translation. We will provide you with all of the
translations produced in the translation task along with the
human reference translations. We are looking for automatic metric scores
for translations at the system-level, document-level, segment-level.
For some languages (English to/from German and Czech, and for English
to Chinese), segments are paragraphs that can contain multiple sentences.
Note that online news text typically has short paragraphs (generally
the average for each reference/source is less than 2 sentences).
We will calculate the system-level, document-level, and
segment(sentence/paragraph)-level correlations of your scores with WMT20
human judgements once the manual evaluation has been completed.
"""

_HOMEPAGE = "https://www.statmt.org/wmt20/metrics-task.html"

_CITATION = """\
@inproceedings{mathur2020results,
  title={Results of the WMT20 metrics shared task},
  author={
    Mathur, Nitika and
    Wei, Johnny and
    Freitag, Markus and
    Ma, Qingsong and
    Bojar, Ond{\v{r}}ej
  },
  booktitle={Proceedings of the Fifth Conference on Machine Translation},
  pages={688--725},
  year={2020}
}
"""

_LANGUAGES = [
    "cs-en",
    "de-en",
    "iu-en",
    "ja-en",
    "km-en",
    "pl-en",
    "ps-en",
    "ru-en",
    "ta-en",
    "zh-en",
    "en-cs",
    "en-de",
    "en-iu",
    "en-ja",
    "en-pl",
    "en-ru",
    "en-ta",
    "en-zh",
]

_SUPPORTED_VERSIONS = [
    "raw",  # raw scores don't filter
    "zscore",  # z scores don't filter
    "raw_nohuman",  # raw scores filter out human systems
    "zscore_nohuman",  # z scores filter out human systems
]


class Wmt20Metrics(datalabs.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name=f"{lang}_{version}",
        )
        for lang in _LANGUAGES
        for version in _SUPPORTED_VERSIONS
    ]

    def _info(self):
        features = datalabs.Features(
            {
                "source": datalabs.Value("string"),
                "references": datalabs.Sequence(datalabs.Value("string")),
                "hypotheses": datalabs.Sequence(
                    {
                        "system_name": datalabs.Value("string"),
                        "hypothesis": datalabs.Value("string"),
                    }
                ),
                "scores": datalabs.Sequence(datalabs.Value("float")),
                "seg_id": datalabs.Value("string"),
            }
        )
        src_lang, trg_lang = self.config.name.split("_")[0].split("-")
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=[src_lang, trg_lang],
            task_templates=[
                get_task(TaskType.meta_evaluation_nlg)(
                    source_column="source",
                    hypotheses_column="hypotheses",
                    references_column="references",
                    scores_column="scores",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        lang = str(self.config.name.split("_")[0])
        dl_paths = dl_manager.download_and_extract(
            {
                "scores": _DL_URLS["scores"] + f"metrics-ad-seg-scores-{lang}.csv",
                "text": _DL_URLS["text"],
            }
        )

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "scores_path": dl_paths["scores"],
                    "text_path": dl_paths["text"],
                },
            ),
        ]

    def _generate_examples(self, scores_path, text_path):
        """Yields examples as (key, example) tuples."""

        text_dir = os.path.join(text_path, "txt")
        config_cols = self.config.name.split("_")
        lang_pair = config_cols[0]
        print(f"{lang_pair=}")
        src_lang, trg_lang = lang_pair.split("-")
        score_type = config_cols[1]
        score_col = "Z.SCR" if score_type == "zscore" else "RAW.SCR"
        filter_human = len(config_cols) > 2

        score_data = {}
        with open(scores_path, "r") as f:
            csvreader = csv.DictReader(f, delimiter=" ")
            for row in csvreader:
                system_name = row["SYS"]
                seg_id = row["SEGID"]
                score = float(row[score_col])
                if seg_id not in score_data:
                    score_data[seg_id] = {}
                score_data[seg_id][system_name] = score

        file_ref = f"newstest2020-{src_lang}{trg_lang}-ref.{trg_lang}.txt"
        with open(os.path.join(text_dir, "references", file_ref), "r") as f:
            refs = [x.strip() for x in f.readlines()]

        file_src = f"newstest2020-{src_lang}{trg_lang}-src.{src_lang}.txt"
        with open(os.path.join(text_dir, "sources", file_src), "r") as f:
            srcs = [x.strip() for x in f.readlines()]

        assert len(srcs) == len(
            refs
        ), f"{lang_pair}, srcs number is different from refs"

        seg_ids = [None for _ in srcs]
        with open(os.path.join(text_dir, "details", f"{lang_pair}.txt"), "r") as f:
            for row in f:
                tmp = row.strip("\n").split("\t")
                seg_ids[int(tmp[0]) - 1] = tmp[3] + "::" + tmp[-1]

        dir_sys = os.path.join(text_dir, "system-outputs", lang_pair)
        file_sys = [
            file for file in os.listdir(dir_sys) if file.startswith("newstest2020")
        ]

        final_data = [
            {
                "source": src,
                "references": list([ref]),
                "hypotheses": list(),
                "scores": list(),
                "seg_id": seg_id,
            }
            for src, ref, seg_id in zip(srcs, refs, seg_ids)
        ]
        for file in file_sys:
            file_cols = file.split(".")
            system_name = f"{file_cols[2]}.{file_cols[3]}"
            if filter_human and "human" in system_name.lower():
                continue
            with open(os.path.join(dir_sys, file), "r") as f:
                sys_data = [x.strip() for x in f.readlines()]
            assert len(sys_data) == len(
                refs
            ), f"language {lang_pair}, {file} number is different from refs"

            for sys_datum, final_datum, seg_id in zip(sys_data, final_data, seg_ids):
                if seg_id in score_data and system_name in score_data[seg_id]:
                    final_datum["hypotheses"].append(
                        {"system_name": system_name, "hypothesis": sys_datum}
                    )
                    final_datum["scores"].append(score_data[seg_id][system_name])

        id = 0
        for x in final_data:
            if x["hypotheses"]:
                sorted_hyp_scores = sorted(
                    [(hyp, score) for hyp, score in zip(x["hypotheses"], x["scores"])],
                    key=lambda x: x[0]["system_name"],
                )
                x["hypotheses"] = [hyp for hyp, _ in sorted_hyp_scores]
                x["scores"] = [score for _, score in sorted_hyp_scores]
                yield id, x
                id += 1
