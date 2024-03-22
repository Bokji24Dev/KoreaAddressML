# KoreaAddressML

한국 주소를 예측 머신러닝 추론 모델입니다.

## Example

```python
from koraddr import KoreanAddress

model = KoreanAddress('model/2024_02')

address = model.predict('서울시 테헤란로25길 13')
print(address)
# 서울특별시

address = model.deeper_predict('역삼동 644-10')
print(address)
# 서울특별시 강남구
```

## Package Install

### CPU

```bash
# 종속성 설치
pip install -U pip setuptools wheel
pip install -U spacy

# 모델 복제
git clone https://huggingface.co/Mineru/ko-address model/2024_02
```

### GPU(NVIDA Only)

```bash
# 종속성 설치
pip install -U pip setuptools wheel
pip install -U 'spacy[cuda11x]'

# 모델 복제
git clone https://huggingface.co/Mineru/ko-address model/2024_02
```

### Apple Silicon(M1)

```bash
# 종속성 설치
pip install -U pip setuptools wheel
pip install -U 'spacy[apple]'

# 모델 복제
git clone https://huggingface.co/Mineru/ko-address model/2024_02
```

## Dataset

- [공개하는 주소](https://business.juso.go.kr/addrlink/attrbDBDwld/attrbDBDwldList.do?cPath=99MD&menu=%EA%B1%B4%EB%AC%BCDB)