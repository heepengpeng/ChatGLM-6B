# {"prompt": "骂我两句", "response": "你算哪块小饼干", "history": []}
import json
import random
import csv

train_list = []
prompt_template = '{"prompt": "%s", "response": "%s", "history": []}'
with open('./ft/data.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|', )
    next(reader)
    for row in reader:
        if len(row) >= 4 and row[3] == "FAQ":
            prompt_template = {"prompt": row[0], "response": row[1], "history": []}
            train_list.append(json.dumps(prompt_template, ensure_ascii=False))

with open('./ft/train_data.json', 'w+', encoding='utf-8') as f:
    for i in train_list:
        f.write("%s\n" % i)

n = len(train_list) // 10
val_list = random.sample(train_list, n)
with open('./ft/val_data.json', 'w+', encoding='utf-8') as f:
    for i in val_list:
        f.write("%s\n" % i)
