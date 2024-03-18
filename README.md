# KoreaAddressML

한국 주소를 예측 머신러닝 추론 모델입니다.

## Example

```python
from koraddr import KoreanAddress

model = KoreanAddress('model/2024_02')

address = model.predict('서울시 테헤란로25길 13')
print(address)
# 서울특별시

address = model.deeper_predict('테헤란로25길 13')
print(address)
# 서울특별시 강남구
```

## Package Install

### CPU

```bash
pip install -U pip setuptools wheel
pip install -U spacy
pip install fire
```

### GPU(NVIDA Only)

```bash
pip install -U pip setuptools wheel
pip install -U 'spacy[cuda11x]'
pip install fire
```

### Apple Silicon(M1)

```bash
pip install -U pip setuptools wheel
pip install -U 'spacy[apple]'
pip install fire
```

## Dataset

- [공개하는 주소](https://business.juso.go.kr/addrlink/attrbDBDwld/attrbDBDwldList.do?cPath=99MD&menu=%EA%B1%B4%EB%AC%BCDB)