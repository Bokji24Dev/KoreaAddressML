# -*- coding:utf-8 -*-
from transformers import LlamaTokenizerFast
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

# 텍스트 데이터 파일 경로
input_file = 'dataset/koraddr.txt'
# 모델 및 보캐블러리 파일을 저장할 경로와 이름
model_prefix = 'tokenizer'
# BPE 알고리즘 사용
model_type = 'bpe'
# 보캐블러리 크기
vocab_size = 13000
# 센텐스피스 트레이너를 사용하여 모델 훈련
SentencePieceTrainer.train(
    f'--input={input_file} ' +
    f'--model_prefix={model_prefix} ' +
    f'--vocab_size={vocab_size} '+
    f'--model_type=bpe ' +
    f'--byte_fallback=true ' +
    f'--max_sentence_length=128'
)

tokenizer = LlamaTokenizerFast(vocab_file=f"{model_prefix}.model")
tokenizer.save_pretrained("tokenizer")

# 훈련된 모델을 로드
sp = SentencePieceProcessor()
sp.load(f'{model_prefix}.model')
