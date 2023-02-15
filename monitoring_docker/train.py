import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
import random
from sklearn.model_selection import train_test_split 
import pandas as pd

#이 부분은 추후 feast로 변경할 것
df = pd.read_csv("dataset.csv",encoding='utf-8')
SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)

#하이퍼파라미터
BATCH_SIZE = 64
lf = 0.001
EPOCHS = 10

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:",DEVICE)

#torchtext.data 의 Field 클래스를 사용하여 메세지 내용에 대한 객체 TEXT, 레이블을 위한 객체 LABEL을 생성
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)
#데이터셋이 순차적인 데이터셋임을 알 수 있도록 sequential 인자값으로 True를 명시해준다. 레이블은 단순한 클래스를 나타내는 숫자로 순차적인 데이터가 아니므로 False를 명시한다.
# batch_first 는 신경망에 입력되는 텐서의 첫번째 차원값이 batch_size가 되도록 한다. 그리고 lower 변수를 통해 텍스트 데이터 속 모든 영문 알파벳이 소문자가 되도록 한다.

#단어 집합 만들기
#min_freq는 학습 데이터에서 최소 5번 이상 등장한 단어만을 단어 집합에 추가하겠다는 의미이다.
#이때 학습 데이터에서 5번 미만으로 등장한 단어는 Unknown이라는 의미에서 '<unk>'라는 토큰으로 대체된다.
TEXT.build_vocab(trainset, min_freq=5)# 단어 집합 생성
LABEL.build_vocab(trainset)

vocab_size = len(TEXT.vocab)
n_classes = 2
print('단어 집합의 크기: {}'.format(vocab_size))
print('클래스의 개수 : {}'.format(n_classes))

#stoi로 단어와 각 단어의 정수 인덱스가 저장되어져 있는 딕셔너리 객체에 접근
print(TEXT.vocab.stoi)

x_train,x_test,y_train,y_test = train_test_split(data.x, data.y, test_size=0.2, random_state=1234)