import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
from sklearn.model_selection import train_test_split
import random

import feature_store_docker.get_data as data

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

x_train,x_test,y_train,y_test = train_test_split(data.x, data.y, test_size=0.2, random_state=1234)