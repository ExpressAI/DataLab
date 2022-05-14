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



_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/iflytek/train.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/iflytek/dev.json"
# _TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/iflytek/test.json"


class IFLYTEK(datalabs.GeneratorBasedBuilder):

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=[
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
                        "其他"
                    ]),
                }
            ),
            homepage="",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[get_task(TaskType.topic_classification)(
                text_column="text",
                label_column="label")],
        )

    def _split_generators(self, dl_manager):
        
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]
        


    def _generate_examples(self, filepath):
        
        
        textualize_label = {
            "0": "打车",
            "1": "地图导航",
            "2": "免费WIFI",
            "3": "租车",
            "4": "同城服务",
            "5": "快递物流",
            "6": "婚庆",
            "7": "家政",
            "8": "公共交通",
            "9": "政务",
            "10": "社区服务",
            "11": "薅羊毛",
            "12": "魔幻",
            "13": "仙侠",
            "14": "卡牌",
            "15": "飞行空战",
            "16": "射击游戏",
            "17": "休闲益智",
            "18": "动作类",
            "19": "体育竞技",
            "20": "棋牌中心",
            "21": "经营养成",
            "22": "策略",
            "23": "MOBA",
            "24": "辅助工具",
            "25": "约会社交",
            "26": "即时通讯",
            "27": "工作社交",
            "28": "论坛圈子",
            "29": "婚恋社交",
            "30": "情侣社交",
            "31": "社交工具",
            "32": "生活社交",
            "33": "微博博客",
            "34": "新闻",
            "35": "漫画",
            "36": "小说",
            "37": "技术",
            "38": "教辅",
            "39": "问答交流",
            "40": "搞笑",
            "41": "杂志",
            "42": "百科",
            "43": "影视娱乐",
            "44": "求职",
            "45": "兼职",
            "46": "视频",
            "47": "短视频",
            "48": "音乐",
            "49": "直播",
            "50": "电台",
            "51": "K歌",
            "52": "成人",
            "53": "中小学",
            "54": "职考",
            "55": "公务员",
            "56": "英语",
            "57": "视频教育",
            "58": "高等教育",
            "59": "成人教育",
            "60": "艺术",
            "61": "语言(非英语)",
            "62": "旅游资讯",
            "63": "综合预定",
            "64": "民航",
            "65": "铁路",
            "66": "酒店",
            "67": "行程管理",
            "68": "民宿短租",
            "69": "出国",
            "70": "工具",
            "71": "亲子儿童",
            "72": "母婴",
            "73": "驾校",
            "74": "违章",
            "75": "汽车咨询",
            "76": "汽车交易",
            "77": "日常养车",
            "78": "行车辅助",
            "79": "租房",
            "80": "买房",
            "81": "装修家居",
            "82": "电子产品",
            "83": "问诊挂号",
            "84": "养生保健",
            "85": "医疗服务",
            "86": "减肥瘦身",
            "87": "美妆美业",
            "88": "菜谱",
            "89": "餐饮店",
            "90": "体育咨讯",
            "91": "运动健身",
            "92": "支付",
            "93": "保险",
            "94": "股票",
            "95": "借贷",
            "96": "理财",
            "97": "彩票",
            "98": "记账",
            "99": "银行",
            "100": "美颜",
            "101": "影像剪辑",
            "102": "摄影修图",
            "103": "相机",
            "104": "绘画",
            "105": "二手",
            "106": "电商",
            "107": "团购",
            "108": "外卖",
            "109": "电影票务",
            "110": "社区超市",
            "111": "购物咨询",
            "112": "笔记",
            "113": "办公",
            "114": "日程管理",
            "115": "女性",
            "116": "经营",
            "117": "收款",
            "118": "其他"
        }
        

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                label, label_des, text = line['label'], line['label_des'], line['sentence']
                if label in textualize_label:
                    label = textualize_label[label]
                    yield id_, {'text': text, 'label': label}

                
                


