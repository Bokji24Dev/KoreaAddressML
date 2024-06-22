# -*- coding:utf-8 -*-
import pickle
import warnings
from typing import Union, List
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

# 공백 토큰
blank_token = 29528


def pad_list_to_32(data: List[int], padding_value=blank_token):
    # 리스트의 길이가 32보다 작은 동안 blank_token 를 추가
    while len(data) < 32:
        data.append(padding_value)
    return data


class KoreanAddress:
    def __init__(self, model: Union[str, None]):
        # 전용 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained("Mineru/KoreaAddressTokenizer")
        if model.__class__ is str:
            # 머신러닝 모델 로드
            with open(model, "rb") as f:
                self.model = pickle.load(f)
        else:
            raise "Errror: model must be str(model file path)"

    def encoding(self, text: str):
        rows = []
        tokens = self.tokenizer.batch_encode_plus([text])
        for ids in tokens["input_ids"]:
            rows.append(pad_list_to_32(ids))
        return rows

    def predict(self, text: str) -> Union[str, None]:
        X = self.encoding(text)
        return self.model.predict(X)


mode = KoreanAddress("dataset/ensemble_model_v1.joblib")
for addr in [
    "부산광역시 사상구 대동로",
    "부산시 사상구 대동로",
    "부산 사상구 대동로",
    "사상구 대동로",
]:
    res = mode.predict(addr)
    print(res)
