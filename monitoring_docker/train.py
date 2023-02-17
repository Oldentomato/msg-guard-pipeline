import torch
import torchtext
from torchtext.legacy import data # torchtext.data 임포트
from torchtext.legacy.data import TabularDataset,BucketIterator
import re
import random
import torch.nn as nn
import torch.nn.functional as F
from konlpy.tag import Okt 
import os
from model import GRU
from joblib import dump

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

okt=Okt() 

#필드 정의
ID = data.Field(sequential = False, use_vocab = False)
TEXT = data.Field(sequential = True,
                  use_vocab = True,
                  tokenize = okt.morphs,
                  lower = True,
                  batch_first = True,
                  fix_length = 20)
LABEL = data.LabelField()

#데이터를 불러와서 데이터셋의 형식으로 바꿔주고, 그와 동시에 토큰화를 수행
all_datas = TabularDataset(
    'result_data.csv', format='csv', fields=[("count",None),('id',ID),('msg_body',TEXT),('category',LABEL)], skip_header=True
)

SEED =1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_data, test_data = all_datas.split(split_ratio=0.8, stratified=False, strata_field = 'label', random_state = random.seed(SEED))

print('훈련 샘플의 갯수: {}'.format(len(train_data)))
print('테스트 샘플의 개수: {}'.format(len(test_data)))

print(vars(train_data[0])) #확인용

#단어 집합 만들기
# min_freq: 단어 집합에 추가 시 단어의 최소 등장 빈도 조건을 추가
# max_size: 단어 집합의 최대 크기를 지정
TEXT.build_vocab(train_data, min_freq=2, max_size=10000)
LABEL.build_vocab(train_data)

print('단어 집합의 크기: {}'.format(len(TEXT.vocab)))
print('라벨의 갯수: {}'.format(len(LABEL.vocab)))

print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.stoi)
print(LABEL.vocab.stoi)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, test_iterator = BucketIterator.splits((train_data, test_data),batch_size = BATCH_SIZE, shuffle=True,sort=False, device = device)

print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_iterator)))
print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_iterator)))


    
model = GRU(1, 256, len(TEXT.vocab), 896, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.msg_body.to(device), batch.category.to(device)
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
        x, y = batch.msg_body.to(device), batch.category.to(device)
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
        best_val_loss = val_loss

#검증
model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iterator)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))