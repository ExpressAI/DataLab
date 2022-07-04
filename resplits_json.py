import random
import requests
import json
import csv

def urldownload(url,filename=None):
    down_res = requests.get(url)
    with open(filename,'wb') as file:
        file.write(down_res.content)

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/dureader_search/train.json"

urldownload(_TRAIN_DOWNLOAD_URL, filename='old_train.json')

# 获取example

with open('old_train.json', encoding="utf-8") as f:
    count = 0
    example_data = open('example_data.json', 'w')

    # 从这里抄之前脚本

    for id_, line in enumerate(f.readlines()):
        line = json.loads(line)
        documents = line["documents"]
        answers, segmented_answers, fake_answers, answer_spans = (
            line["answers"],
            line["segmented_answers"],
            line["fake_answers"],
            line["answer_spans"],
        )
        (
            question,
            segmented_question,
            question_type,
            fact_or_opinion,
            question_id,
        ) = (
            line["question"],
            line["segmented_question"],
            line["question_type"],
            line["fact_or_opinion"],
            line["question_id"],
        )
        match_scores, answer_docs = line["match_scores"], line["answer_docs"]
        line_data = {
            "documents": documents,
            "answers": answers,
            "segmented_answers": segmented_answers,
            "fake_answers": fake_answers,
            "answer_spans": answer_spans,
            "question": question,
            "segmented_question": segmented_question,
            "question_type": question_type,
            "fact_or_opinion": fact_or_opinion,
            "question_id": question_id,
            "match_scores": match_scores,
            "answer_docs": answer_docs,
        }

        line_data = json.dumps(line_data, ensure_ascii=False)
        example_data.write(line_data + '\n')
        count = count + 1


    example_data.close()
    print('example_num: ', count) 


# 生成空的新文件
train_data = open('train_revised_search.json', 'w')
dev_data = open('validation_revised.json', 'w')
test_data = open('test_revised_search.json', 'w')

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
    dev_data.close()
    test_data.close()
    print('done')


# 查看新的train里面是不是我要的数目
with open('train_revised_search.json', encoding="utf-8") as f:
    lines = f.readlines()
    train_num = len(lines)
    print("train_num: ", train_num)  # 确认数目是不是和想要的一样


# dev
with open('validation_revised.json', encoding="utf-8") as f:
    lines = f.readlines()
    dev_num = len(lines)
    print("dev_num: ", dev_num)  # 确认数目是不是和想要的一样

# test
with open('test_revised_search.json', encoding="utf-8") as f:
    lines = f.readlines()
    test_num = len(lines)
    print("test_num: ", test_num)  # 确认数目是不是和想要的一样

'''
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/dureader_zhidao/dev.json"

urldownload(_VALIDATION_DOWNLOAD_URL, filename='old_dev.json')

# 获取example

with open('old_dev.json', encoding="utf-8") as f:
    count = 0
    dev_data = open('validation_revised.json', 'w')  #打开文件，这里记得改

    # 从这里抄之前脚本
    for id_, line in enumerate(f.readlines()):
        line = json.loads(line)
        documents = line["documents"]
        answers, segmented_answers, fake_answers, answer_spans = (
            line["answers"],
            line["segmented_answers"],
            line["fake_answers"],
            line["answer_spans"],
        )
        (
            question,
            segmented_question,
            question_type,
            fact_or_opinion,
            question_id,
        ) = (
            line["question"],
            line["segmented_question"],
            line["question_type"],
            line["fact_or_opinion"],
            line["question_id"],
        )
        match_scores, answer_docs = line["match_scores"], line["answer_docs"]
        line_data = {
            "documents": documents,
            "answers": answers,
            "segmented_answers": segmented_answers,
            "fake_answers": fake_answers,
            "answer_spans": answer_spans,
            "question": question,
            "segmented_question": segmented_question,
            "question_type": question_type,
            "fact_or_opinion": fact_or_opinion,
            "question_id": question_id,
            "match_scores": match_scores,
            "answer_docs": answer_docs,
        }


        line_data = json.dumps(line_data, ensure_ascii=False)
        dev_data.write(line_data + '\n')
        count = count + 1


    dev_data.close()
    print('dev_num: ', count) 


for id_, line in enumerate(f.readlines()):
    line = json.loads(line.strip())
                
'''