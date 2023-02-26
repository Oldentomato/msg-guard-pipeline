import torch
import torchtext
from torchtext.legacy.data import BucketIterator
import random
import os
from model import GRU
from joblib import dump
from data_preprocessing import MsgTrainModel
import os
from datetime import datetime


SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)

#하이퍼파라미터
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 100

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:",device)



best_val_loss = None

model_cls = MsgTrainModel("/code/datas/","msg_svc",BATCH_SIZE, device)

train_iterator,test_iterator,text_len = model_cls.get_training_data()

model = GRU(1, 256, text_len, 896, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for e in range(1, EPOCHS+1):
    model_cls.train_model(model,optimizer, train_iterator)
    val_loss, val_accuracy = model_cls.evaluate_model(model,test_iterator)

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        if not os.path.isdir("artifacts"):
            os.makedirs("artifacts")
        dump(model, "artifacts/model.joblib")
        torch.save(model.state_dict(), './snapshot/'+datetime.today().strftime("%Y/%m/%d %H:%M:%S")+'.pt')
        best_val_loss = val_loss


#검증
model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = model_cls.evaluate_model(model, test_iterator)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))