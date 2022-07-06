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
from datalabs import get_task, TaskType

_DESCRIPTION = """\
Iflytek is a long text data set about descriptions of the application softwares, containing a variety of topics related to daily life.
Each text is labelled with one of 119 categories. For more information, please refer https://github.com/CLUEbenchmark/CLUE. 
"""

_CITATION = """\
@article{xu2020clue,
  title={CLUE: A Chinese language understanding evaluation benchmark},
  author={Xu, Liang and Hu, Hai and Zhang, Xuanwei and Li, Lu and Cao, Chenjie and Li, Yudong and Xu, Yechen and Sun, Kai and Yu, Dian and Yu, Cong and others},
  journal={arXiv preprint arXiv:2004.05986},
  year={2020}
}
"""


_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/iflytek/train_revised.json"
_VALIDATION_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/iflytek/validation_revised.json"
)
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/iflytek/test_revised.json"


class IFLYTEK(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=[
                            "打车",
                            "地图导航",
                            "免费WIFI",
                            "租车",
                            "同城服务",
                            "快递物流",
                            "婚庆",
                            "家政",
                            "公共交通",
                            "政务",
                            "社区服务",
                            "薅羊毛",
                            "魔幻",
                            "仙侠",
                            "卡牌",
                            "飞行空战",
                            "射击游戏",
                            "休闲益智",
                            "动作类",
                            "体育竞技",
                            "棋牌中心",
                            "经营养成",
                            "策略",
                            "MOBA",
                            "辅助工具",
                            "约会社交",
                            "即时通讯",
                            "工作社交",
                            "论坛圈子",
                            "婚恋社交",
                            "情侣社交",
                            "社交工具",
                            "生活社交",
                            "微博博客",
                            "新闻",
                            "漫画",
                            "小说",
                            "技术",
                            "教辅",
                            "问答交流",
                            "搞笑",
                            "杂志",
                            "百科",
                            "影视娱乐",
                            "求职",
                            "兼职",
                            "视频",
                            "短视频",
                            "音乐",
                            "直播",
                            "电台",
                            "K歌",
                            "成人",
                            "中小学",
                            "职考",
                            "公务员",
                            "英语",
                            "视频教育",
                            "高等教育",
                            "成人教育",
                            "艺术",
                            "语言(非英语)",
                            "旅游资讯",
                            "综合预定",
                            "民航",
                            "铁路",
                            "酒店",
                            "行程管理",
                            "民宿短租",
                            "出国",
                            "工具",
                            "亲子儿童",
                            "母婴",
                            "驾校",
                            "违章",
                            "汽车咨询",
                            "汽车交易",
                            "日常养车",
                            "行车辅助",
                            "租房",
                            "买房",
                            "装修家居",
                            "电子产品",
                            "问诊挂号",
                            "养生保健",
                            "医疗服务",
                            "减肥瘦身",
                            "美妆美业",
                            "菜谱",
                            "餐饮店",
                            "体育咨讯",
                            "运动健身",
                            "支付",
                            "保险",
                            "股票",
                            "借贷",
                            "理财",
                            "彩票",
                            "记账",
                            "银行",
                            "美颜",
                            "影像剪辑",
                            "摄影修图",
                            "相机",
                            "绘画",
                            "二手",
                            "电商",
                            "团购",
                            "外卖",
                            "电影票务",
                            "社区超市",
                            "购物咨询",
                            "笔记",
                            "办公",
                            "日程管理",
                            "女性",
                            "经营",
                            "收款",
                            "其他",
                        ]
                    ),
                }
            ),
            homepage="https://github.com/CLUEbenchmark/CLUE",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.topic_classification)(
                    text_column="text", label_column="label"
                )
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}
            ),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())            
                yield id_, {"text": line["text"], "label": line["label"]}
