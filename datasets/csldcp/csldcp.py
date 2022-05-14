# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import datalabs
from datalabs.tasks import TopicClassification


_DESCRIPTION = """\
CSLDCP: Chinese Scientific Literature Dataset for Classification of Projects
This dataset contains 67 literature topics from 13 categories, ranging from social sciences to natural sciences, with text in Chinese abstracts.
Data scale: train set(536), validation set(536), test set with labels(1784), test set without labels(2999), unlabeled corpus(67). 
For more information, please refer to https://github.com/CLUEbenchmark/FewCLUE. 
"""

_CITATION = """\
@article{Xu2021FewCLUEAC,
  title={FewCLUE: A Chinese Few-shot Learning Evaluation Benchmark},
  author={Liang Xu and Xiaojing Lu and Chenyang Yuan and Xuanwei Zhang and Huining Yuan and Huilin Xu and Guoao Wei and Xiang Pan and Hai Hu},
  journal={ArXiv},
  year={2021},
  volume={abs/2107.07498}
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/csldcp/train.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/csldcp/dev.json"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/csldcp/test.json"
# _TEST_UNLABELED_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/csldcp/test_unlabeled.json"
# _UNLABELED_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/csldcp/unlabeled.json"

class CSLDCP(datalabs.GeneratorBasedBuilder):
    
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=[
                        '材料', 
                        '作物', 
                        '口腔', 
                        '药学', 
                        '教育', 
                        '水利', 
                        '理经', 
                        '食品', 
                        '兽医', 
                        '体育', 
                        '核能', 
                        '力学', 
                        '园艺', 
                        '水产', 
                        '法学', 
                        '地质', 
                        '能源', 
                        '农林', 
                        '通信', 
                        '情报', 
                        '政治', 
                        '电气', 
                        '海洋', 
                        '民族', 
                        '航空', 
                        '化工', 
                        '哲学', 
                        '卫生', 
                        '艺术', 
                        '农工', 
                        '船舶', 
                        '计科', 
                        '冶金', 
                        '交通', 
                        '动力', 
                        '纺织', 
                        '建筑', 
                        '环境', 
                        '公管', 
                        '数学', 
                        '物理', 
                        '林业', 
                        '心理', 
                        '历史', 
                        '工商', 
                        '应经', 
                        '中医', 
                        '天文', 
                        '机械', 
                        '土木', 
                        '光学', 
                        '地理', 
                        '农资', 
                        '生物', 
                        '兵器', 
                        '矿业', 
                        '大气', 
                        '医学', 
                        '电子', 
                        '测绘', 
                        '控制', 
                        '军事', 
                        '语言', 
                        '新闻', 
                        '社会', 
                        '地球', 
                        '植物'
                    ]),
                }
            ),
            homepage="https://github.com/CLUEbenchmark/FewCLUE",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[TopicClassification(text_column="text", label_column="label", task="topic-classification")],
        )

    def _split_generators(self, dl_manager):
        
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]
        


    def _generate_examples(self, filepath):
        
        
        label_des2tag={
            "材料科学与工程":"材料",
            "作物学":"作物",
            "口腔医学":"口腔",
            "药学":"药学",
            "教育学":"教育",
            "水利工程":"水利",
            "理论经济学":"理经",
            "食品科学与工程":"食品",
            "畜牧学/兽医学":"兽医",
            "体育学":"体育",
            "核科学与技术":"核能",
            "力学":"力学",
            "园艺学":"园艺",
            "水产":"水产",
            "法学":"法学",
            "地质学/地质资源与地质工程":"地质",
            "石油与天然气工程":"能源",
            "农林经济管理":"农林",
            "信息与通信工程":"通信",
            "图书馆、情报与档案管理":"情报",
            "政治学":"政治",
            "电气工程":"电气",
            "海洋科学":"海洋",
            "民族学":"民族",
            "航空宇航科学与技术":"航空",
            "化学/化学工程与技术":"化工",
            "哲学":"哲学",
            "公共卫生与预防医学":"卫生",
            "艺术学":"艺术",
            "农业工程":"农工",
            "船舶与海洋工程":"船舶",
            "计算机科学与技术":"计科",
            "冶金工程":"冶金",
            "交通运输工程":"交通",
            "动力工程及工程热物理":"动力",
            "纺织科学与工程":"纺织",
            "建筑学":"建筑",
            "环境科学与工程":"环境",
            "公共管理":"公管",
            "数学":"数学",
            "物理学":"物理",
            "林学/林业工程":"林业",
            "心理学":"心理",
            "历史学":"历史",
            "工商管理":"工商",
            "应用经济学":"应经",
            "中医学/中药学":"中医",
            "天文学":"天文",
            "机械工程":"机械",
            "土木工程":"土木",
            "光学工程":"光学",
            "地理学":"地理",
            "农业资源利用":"农资",
            "生物学/生物科学与工程":"生物",
            "兵器科学与技术":"兵器",
            "矿业工程":"矿业",
            "大气科学":"大气",
            "基础医学/临床医学":"医学",
            "电子科学与技术":"电子",
            "测绘科学与技术":"测绘",
            "控制科学与工程":"控制",
            "军事学":"军事",
            "中国语言文学":"语言",
            "新闻传播学":"新闻",
            "社会学":"社会",
            "地球物理学":"地球",
            "植物保护":"植物"
            }
        

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                text, label = line['content'], line['label']
                if label in label_des2tag:
                    label = label_des2tag[label]
                    yield id_, {'text': text, 'label': label}

                
                


