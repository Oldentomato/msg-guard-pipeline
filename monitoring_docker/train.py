import torch
import torchtext
from torchtext.legacy import data # torchtext.data 임포트
from torchtext.legacy.data import BucketIterator
import re
import random
import torch.nn as nn
import torch.nn.functional as F
from konlpy.tag import Okt 
import os
from model import GRU
from joblib import dump
import logging
from data_preprocessing import Create_DataSet

#이 부분은 추후 feast로 변경할 것
df = pd.read_csv("dataset.csv",encoding='utf-8')
SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)

#하이퍼파라미터
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 100

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:",DEVICE)

train_data, test_data, text_len = Create_DataSet(df)#df 는 임시 feast에서 받아오는걸로 변경할 것


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, test_iterator = BucketIterator.splits((train_data, test_data),batch_size = BATCH_SIZE, shuffle=True,sort=False, device = DEVICE)

print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_iterator)))
print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_iterator)))


    
model = GRU(1, 256, text_len, 896, 2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.msg_body.to(DEVICE), batch.category.to(DEVICE)
#         y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()


def evaluate(model, val_iter):
    """evaluate model"""
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.msg_body.to(DEVICE), batch.category.to(DEVICE)
#         y.data.sub_(1) # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

#학습
best_val_loss = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_iterator)
    val_loss, val_accuracy = evaluate(model, test_iterator)

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        dump(model, "artifacts/model.joblib")
        best_val_loss = val_loss

#검증
model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iterator)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))