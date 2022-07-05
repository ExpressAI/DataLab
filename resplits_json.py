import random
import requests
import json
import csv

def urldownload(url,filename=None):
    down_res = requests.get(url)
    with open(filename,'wb') as file:
        file.write(down_res.content)

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/tnews/train.json"

urldownload(_TRAIN_DOWNLOAD_URL, filename='old_train.json')

textualize_label = {
            "100": "story",
            "101": "culture",
            "102": "entertainment",
            "103": "sports",
            "104": "finance",
            "106": "house",
            "107": "car",
            "108": "edu",
            "109": "tech",
            "110": "military",
            "112": "travel",
            "113": "world",
            "114": "stock",
            "115": "agriculture",
            "116": "game",
        }

# 获取example

with open('old_train.json', encoding="utf-8") as f:
    count = 0
    example_data = open('example_data.json', 'w')

    # 从这里抄之前脚本

    for line in f:
        res_info = json.loads(line)
        if res_info.__contains__("label"):
            label = textualize_label[res_info["label"]]
            line_data = {
                "text": res_info["sentence"],
                "keywords": res_info["keywords"],
                "label": label,
            }
    
    #到这里

            line_data = json.dumps(line_data, ensure_ascii=False)
            example_data.write(line_data + '\n')
            count = count + 1


    example_data.close()
    print('example_num: ', count) 


# 生成空的新文件
train_data = open('train_revised.json', 'w')
test_data = open('test_revised.json', 'w')

# 把原来的数据集打乱顺序
with open('example_data.json', encoding="utf-8") as f:
    count = 0
    lines = f.readlines()
    random.shuffle(lines)
    for id_, line in enumerate(lines):
        if count < 10000:
            test_data.writelines(line)
        # elif count < 12831:
            # dev_data.writelines(line)
        else:
            train_data.writelines(line)
        count = count + 1
    
    train_data.close()
    test_data.close()
    print('done')

# 查看新的train里面是不是我要的数目
with open('train_revised.json', encoding="utf-8") as f:
    lines = f.readlines()
    train_num = len(lines)
    print("train_num: ", train_num)  # 确认数目是不是和想要的一样


_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/tnews/dev.json"

urldownload(_VALIDATION_DOWNLOAD_URL, filename='old_dev.json')

# 获取example

with open('old_dev.json', encoding="utf-8") as f:
    count = 0
    dev_data = open('validation_revised.json', 'w')  #打开文件，这里记得改

    # 从这里抄之前脚本
    for line in f:
        res_info = json.loads(line)
        if res_info.__contains__("label"):
            label = textualize_label[res_info["label"]]
            line_data = {
                "text": res_info["sentence"],
                "keywords": res_info["keywords"],
                "label": label,
            }
    #到这里

            line_data = json.dumps(line_data, ensure_ascii=False)
            dev_data.write(line_data + '\n')
            count = count + 1


    dev_data.close()
    print('dev_num: ', count) 


# test
with open('test_revised.json', encoding="utf-8") as f:
    lines = f.readlines()
    test_num = len(lines)
    print("test_num: ", test_num)  # 确认数目是不是和想要的一样


'''
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())

# 统计dev的数目
with open('validation_revised.json', encoding="utf-8") as f:
    lines = f.readlines()
    dev_num = len(lines)
    print("dev_num: ", dev_num)  # 确认数目是不是和想要的一样

'''