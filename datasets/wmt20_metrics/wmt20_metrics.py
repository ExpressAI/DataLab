"""WMT20 neural machine translation metrics dataset."""
import os
import csv
import pickle

import datalabs
logger = datalabs.logging.get_logger(__name__)

_DL_URLS = {
    "DArr_seglevel": "https://raw.githubusercontent.com/WMT-Metrics-task/wmt20-metrics/main/manual-evaluation/DArr-seglevel.csv",
    "txt": "https://drive.google.com/uc?export=download&confirm=t&id=1P-Y1P-GTMCNtWj8qaeq-U-m-0DGGnOaP"
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

class Wmt20Metrics(datalabs.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name="{}".format(lang),
        )
        for lang in _LANGUAGES
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "src": datalabs.Value("string"),
                    "ref": datalabs.Value("string"),
                    "better_sys": datalabs.Value("string"),
                    "better_sys_name": datalabs.Value("string"),
                    "worse_sys": datalabs.Value("string"),
                    "worse_sys_name": datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            #task_templates=[
            #],
        )

    def _split_generators(self, dl_manager):
        lang = str(self.config.name)
        dl_paths = dl_manager.download_and_extract(_DL_URLS)
        print(dl_paths)
        data_dir = os.path.join(os.path.dirname(dl_paths['txt']),"wmt20_metrics")

        if not os.path.exists(data_dir) or not lang+'_data.pkl' in os.listdir(data_dir):
            _write_file(dl_paths, data_dir, lang)

        # Generate shared vocabulary
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang+'_data.pkl'),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples as (key, example) tuples."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            for idx_ in data:
                yield idx_, {
                    "src": data[idx_]["src"],
                    "ref": data[idx_]["ref"],
                    "better_sys": data[idx_]["better"]["sys"],
                    "better_sys_name": data[idx_]["better"]["sys_name"],
                    "worse_sys": data[idx_]["worse"]["sys"],
                    "worse_sys_name": data[idx_]["worse"]["sys_name"],
                }

def _readfile(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
    return lines

def _save_pickle(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {file}.')


def _write_file(dl_paths, data_dir, pair):
    file_manual_name = dl_paths['DArr_seglevel']
    dir_documents = dl_paths['txt']
    
    THRESHOLD = 25

    file_manual = open(file_manual_name)
    csvreader = csv.reader(file_manual,delimiter=' ')
    header = next(csvreader)
    LP = [] 
    DATA = []
    SID = [] 
    BETTER = [] 
    WORSE = []
    POINT = []

    for row in csvreader:
        LP.append(row[header.index('LP')])
        DATA.append(row[header.index('DATA')])
        SID.append(row[header.index('SID')])
        BETTER.append(row[header.index('BETTER')])
        WORSE.append(row[header.index('WORSE')])
        POINT.append(float(row[-1]))


    output_data = {}
    
    doc_id = 0
    read_document = True
    errors = set()
    for i in range(len(LP)):
        if LP[i] != pair or DATA[i] != 'newstest2020' or POINT[i] < THRESHOLD:
            continue
        if read_document:
            refs = []
            dir_ref = os.path.join(os.path.join(dir_documents,'txt'),'references')
            file_ref = DATA[i] + '-' + pair.split('-')[0] + pair.split('-')[1] + '-ref.' + pair.split('-')[1] + '.txt'
            refs = _readfile(os.path.join(dir_ref,file_ref))
            num = len(refs)

            srcs = []
            dir_src = os.path.join(os.path.join(dir_documents,'txt'),'sources')
            file_src = DATA[i] + '-' + pair.split('-')[0] + pair.split('-')[1] + '-src.' + pair.split('-')[0] + '.txt'
            srcs = _readfile(os.path.join(dir_src,file_src))
            assert len(srcs)==num, 'language {}, srcs number is different from refs'.format(pair)

            syss = {}
            dir_sys = os.path.join(os.path.join(os.path.join(dir_documents,'txt'),'system-outputs'),LP[i])
            file_sys = [file for file in os.listdir(dir_sys) if file.startswith('newstest2020')]
            for file in file_sys:
                syss[file] = _readfile(os.path.join(dir_sys, file))
                assert len(syss[file])==num, 'language {}, {} number is different form refs'.format(pair, file)

            details_id_SID = {}
            details_SID_id = {}
            dir_details = os.path.join(os.path.join(dir_documents,'txt'),'details')
            dir_details = os.path.join(os.path.join(dir_documents,'txt'),'details')
            file_details = pair + '.txt'
            details = _readfile(os.path.join(dir_details,file_details))
            for row in details:
                tmp = row.split('\t')
                details_id_SID[tmp[0]] = tmp[3]+'::'+tmp[-1][:-1]
                details_SID_id[tmp[3]+'::'+tmp[-1][:-1]] = tmp[0]

            read_document = False
        
        id = int(details_SID_id[SID[i]])-1
        if DATA[i]+'.'+pair+'.'+BETTER[i]+'.txt' in syss and DATA[i]+'.'+pair+'.'+WORSE[i]+'.txt' in syss:
            if doc_id not in output_data:
                output_data[doc_id] = {}
                output_data[doc_id]['better'] = {}
                output_data[doc_id]['worse'] = {}
            output_data[doc_id]['better']['sys'] = syss[DATA[i]+'.'+pair+'.'+BETTER[i]+'.txt'][id]
            output_data[doc_id]['better']['sys_name'] = BETTER[i]
            output_data[doc_id]['better']['scores'] = {}
            output_data[doc_id]['worse']['sys'] = syss[DATA[i]+'.'+pair+'.'+WORSE[i]+'.txt'][id]
            output_data[doc_id]['worse']['sys_name'] = WORSE[i]
            output_data[doc_id]['worse']['scores'] = {}
            output_data[doc_id]['src'] = srcs[id]
            output_data[doc_id]['ref'] = refs[id]
        else:
            errors.add(DATA[i]+'.'+pair+'.'+BETTER[id]+'.txt')
        doc_id += 1
    for err in errors:
        print("{} cannot find {} in syss.".format(pair, err))
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    _save_pickle(output_data, os.path.join(data_dir,pair+'_data.pkl'))
    print("language:{}, DA pairs:{}".format(pair, len(output_data.keys())))
