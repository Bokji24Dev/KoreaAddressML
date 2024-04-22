# -*- coding:utf-8 -*-
import csv
import unittest
from koraddr.interface import KoreanAddress

class KoreanAddressTests(unittest.TestCase):
    def load_dataset(self):
        with open('dataset/road_address.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.road_address.append(row)
        
        with open('dataset/jibun_address.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.jibun_address.append(row)
        
        
    def setUp(self):
        self.model = KoreanAddress("model/2024_04_03", use_gpu=False)
        self.jibun_address = []
        self.road_address = []
        self.load_dataset()


    def test_predict_jibun_address(self):
        for row in self.jibun_address:
            predict = self.model.predict(row['text'])
            print(row['address_1'], "/", predict)
            # assert row['address_1'] == predict


    def test_predict_road_address(self):
        for row in self.road_address:
            predict = self.model.predict(row['text'])
            print(row['address_1'], "/", predict)
            # assert row['address_1'] == predict


    def test_deeper_predict_jibun_address(self):
        for row in self.jibun_address:
            predict = self.model.deeper_predict(row['text'])
            print(row['address_2'], "/", predict)
            # assert row['address_2'] == predict


    def test_deeper_predict_road_address(self):
        for row in self.road_address:
            predict = self.model.deeper_predict(row['text'])
            print(row['address_2'], "/", predict)
            # assert row['address_2'] == predict


if __name__ == '__main__':
    unittest.main()