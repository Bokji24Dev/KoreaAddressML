# -*- coding:utf-8 -*-
import pickle
import warnings
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
blank_token = 29528


def process_string(s):
    # 문자열을 공백을 기준으로 분리
    words = s.split()

    # 단어의 개수가 1개 이하인 경우 변환 없이 반환
    if len(words) <= 1:
        return s

    # 마지막 두 단어가 같은 경우
    if words[-1] == words[-2]:
        # 마지막 단어를 제외하고 나머지를 재조합
        return " ".join(words[:-1])
    else:
        # 두 단어가 같지 않다면 원래 문자열 반환
        return s


def pad_list_to_32(data: List[int], padding_value=blank_token):
    while len(data) < 32:
        data.append(padding_value)
    return data


tokenizer = AutoTokenizer.from_pretrained("Mineru/KoreaAddressTokenizer")

with open("dataset/koraddr_dataset_v2.pk", "rb") as f:
    rows = pickle.load(f)

check_sum = True
idx = 0
texts = []
for i, row in tqdm(enumerate(rows), total=len(rows)):
    row["text"] = process_string(row["text"])
    texts.append(row["text"])
    if check_sum:
        idx = i
        check_sum = False
    if len(texts) == 1024:
        tokens = tokenizer.batch_encode_plus(texts)
        for j, ids in enumerate(tokens["input_ids"]):
            rows[idx + j]["input_ids"] = pad_list_to_32(ids)
        check_sum = True
        texts = []
tokens = tokenizer.batch_encode_plus(texts)
for j, ids in enumerate(tokens["input_ids"]):
    rows[idx + j]["input_ids"] = ids

with open("dataset/koraddr_dataset_v3.pk", "wb") as f:
    pickle.dump(rows, f)
