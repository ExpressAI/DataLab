"""WMT20 neural machine translation metrics dataset."""
import csv
import os
import pickle

import datalabs
from datalabs import get_task, TaskType

logger = datalabs.logging.get_logger(__name__)

_DL_URLS = {
    "DA": "https://raw.githubusercontent.com/WMT-Metrics-task/wmt20-metrics/main/manual-evaluation/DA/",
    "txt": "https://drive.google.com/uc?export=download&confirm=t&id=1P-Y1P-GTMCNtWj8qaeq-U-m-0DGGnOaP",
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
  author={Mathur, Nitika and Wei, Johnny and Freitag, Markus and Ma, Qingsong and Bojar, Ond{\v{r}}ej},
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
    "1.0.0", # don't filter
    "1.0.1", # filter out systems without manual scores
    "1.0.2", # filter out human systems
    "1.0.3", # filter out segments without manual scores
]

class Wmt20Metrics(datalabs.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name="{}_{}".format(lang, version), version=datalabs.Version(version)
        )
        for lang in _LANGUAGES for version in _SUPPORTED_VERSIONS
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "sys_name": datalabs.Value("string"),
                    "seg_id": datalabs.Value("string"),
                    "test_set": datalabs.Value("string"),
                    "src": datalabs.Value("string"),
                    "ref": datalabs.Value("string"),
                    "sys": datalabs.Value("string"),
                    "manual_score_raw": datalabs.Value("string"),
                    "manual_score_z": datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates = [get_task(TaskType.meta_evaluation_wmt)(
                sys_name_column= "sys_name",
                seg_id_column= "seg_id",
                test_set_column = "test_set",
                source_column = "src",
                reference_column = "ref",
                hypothesis_column = "sys",
                manual_score_raw_column = "manual_score_raw",
                manual_score_z_column = "manual_score_z",
            )],
        )

    def _split_generators(self, dl_manager):
        lang = str(self.config.name.split("_")[0])
        version = str(self.config.name.split("_")[1])

        dl_paths = dl_manager.download_and_extract({
                "DA": _DL_URLS["DA"] + 'metrics-ad-seg-scores-{}.csv'.format(lang),
                "txt": _DL_URLS["txt"],
                }
            )
        print(dl_paths)
        data_dir = os.path.join(os.path.dirname(dl_paths["txt"]), "wmt20_metrics")

        if not os.path.exists(data_dir) or not "{}_{}_data.pkl".format(lang, version) in os.listdir(
            data_dir
        ):
            _write_file(dl_paths, data_dir, lang, version)

        # Generate shared vocabulary
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "{}_{}_data.pkl".format(lang, version)),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples as (key, example) tuples."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            for idx_ in data:
                yield idx_, {
                    "sys_name": data[idx_][0],
                    "seg_id": data[idx_][1],
                    "test_set": data[idx_][2],
                    "src": data[idx_][3],
                    "ref": data[idx_][4],
                    "sys": data[idx_][5],
                    "manual_score_raw": data[idx_][6],
                    "manual_score_z": data[idx_][7],
                }


def _readfile(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return lines


def _save_pickle(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {file}.")


def _write_file(dl_paths, data_dir, pair, version):
    file_manual_name = dl_paths["DA"]
    dir_documents = dl_paths["txt"]

    TestSet = 'newstest2020'

    file_manual = open(file_manual_name)
    csvreader = csv.reader(file_manual,delimiter=' ')
    header = next(csvreader)
    manual = {}
    for row in csvreader:
        systemName = row[header.index('SYS')]
        segmentId = row[header.index('SEGID')]
        rawScore = row[header.index('RAW.SCR')]
        ZScore = row[header.index('Z.SCR')]
        if systemName not in manual:
            manual[systemName] = {}
        manual[systemName][segmentId] = (float(rawScore), float(ZScore))

    refs = []
    dir_ref = os.path.join(os.path.join(dir_documents,"txt"),'references')
    file_ref = TestSet + '-' + pair.split('-')[0] + pair.split('-')[1] + '-ref.' + pair.split('-')[1] + '.txt'
    refs = _readfile(os.path.join(dir_ref,file_ref))
    num = len(refs)


    srcs = []
    dir_src = os.path.join(os.path.join(dir_documents,"txt"),'sources')
    file_src = TestSet + '-' + pair.split('-')[0] + pair.split('-')[1] + '-src.' + pair.split('-')[0] + '.txt'
    srcs = _readfile(os.path.join(dir_src,file_src))
    assert len(srcs)==num, 'language {}, srcs number is different from refs'.format(pair)

    syss = {}
    dir_sys = os.path.join(os.path.join(os.path.join(dir_documents,"txt"),'system-outputs'),pair)
    file_sys = [file for file in os.listdir(dir_sys) if file.startswith(TestSet)]
    for file in file_sys:
        systemName = file.split('.')[2]+'.'+file.split('.')[3]
        syss[systemName] = _readfile(os.path.join(dir_sys, file))
        assert len(syss[systemName])==num, 'language {}, {} number is different form refs'.format(pair, file)

    details_id_SID = {}
    details_SID_id = {}
    dir_details = os.path.join(os.path.join(dir_documents,"txt"),'details')
    file_details = pair + '.txt'
    details = _readfile(os.path.join(dir_details,file_details))
    for row in details:
        tmp = row.split('\t')
        details_id_SID[int(tmp[0])] = tmp[3]+'::'+tmp[-1][:-1]
        details_SID_id[tmp[3]+'::'+tmp[-1][:-1]] = int(tmp[0])

    output_data = {}
    index = 0
    for SYSName in sorted(syss.keys()):
        if version == "1.0.1" and SYSName not in manual:
            continue
        if version == "1.0.2" and 'human' in SYSName.lower():
            continue
        for SEGID in sorted(details_SID_id.keys()):
            i = details_SID_id[SEGID] - 1
            src = srcs[i][:-1]
            ref = refs[i][:-1]
            sys = syss[SYSName][i][:-1]
            if version == "1.0.3" and (SYSName not in manual or SEGID not in manual[SYSName]):
                continue
            manualRaw = str(manual[SYSName][SEGID][0]) if (SYSName in manual and SEGID in manual[SYSName]) else ""
            manualZ = str(manual[SYSName][SEGID][1]) if (SYSName in manual and SEGID in manual[SYSName]) else ""

            output_data[index] = [SYSName, SEGID, TestSet, src, ref, sys, manualRaw, manualZ]
            index += 1

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    _save_pickle(output_data, os.path.join(data_dir, "{}_{}_data.pkl".format(pair, version)))
    print("language:{}, version:{} number of examples:{}".format(pair, version, len(output_data.keys())))
