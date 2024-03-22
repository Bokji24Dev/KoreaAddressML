# -*- coding:utf-8 -*-
import spacy
from typing import Union
from spacy import Language

class KoreanAddress:
    def __init__(self, model: Union[str, Language], use_gpu: bool = False):
        if use_gpu:
            spacy.require_gpu()
        if model.__class__ is str:
            self.model = spacy.load(model)
        elif model.__class__ is Language:
            self.model = model
        else:
            raise "Errror: model must be either str or Language"
    
    def predict(self, text: str, threshold: float = 0.2) -> Union[str, None]:
        doc = self.model(text)
        doc = doc.to_json()
        predict = list(doc["cats"].items())
        predict = list(filter(lambda x: x[1] > threshold, predict))
        predict = list(map(lambda x: x[0], predict))
        if len(predict) > 0:
            return predict[0]
        else:
            return None


    def deeper_predict(self, text: str, threshold: float = 0.2) -> Union[str, None]:
        doc = self.model(text)
        doc = doc.to_json()
        predict = list(doc["cats"].items())
        predict = list(filter(lambda x: x[1] > threshold, predict))
        predict = list(map(lambda x: x[0], predict))
        predict.sort(key=lambda x: len(x), reverse=True)
        if len(predict) > 0:
            return predict[0]
        else:
            return None
