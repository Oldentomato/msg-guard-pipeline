import torch
import torchtext
from torchtext.legacy.data import Field, Dataset, Example,LabelField, BucketIterator
from konlpy.tag import Okt 
import pandas as pd
from feast import FeatureStore
from datetime import datetime
import torch.nn.functional as F
import feast
import pickle
from get_data import SetOnlineCleanData


class DataFrameDataset(Dataset):
    """Class for using pandas DataFrames as a datasource"""
    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
            examples pd.DataFrame: DataFrame of examples
            fields {str: Field}: The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): use only exanples for which
                filter_pred(example) is true, or use all examples if None.
                Default is None
        """
        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

class SeriesExample(Example):
    """Class to convert a pandas Series to an Example"""

    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        
        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex
    
def Split_DataFrame(df, frac, random_state):
    train = df.sample(frac=frac, random_state=random_state)
    test = df.drop(train.index)

    return train, test

class MsgPredict:
    def __init__(self,msg_body):
        msg_df = pd.DataFrame(msg_body,index = [0])
        self._msg_df = SetOnlineCleanData(msg_df)

    def SetData(self):
        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if USE_CUDA else "cpu")
        print(self._msg_df)
        if self._msg_df != None: #(광고) 문자열 있는지 체크
            with open('/app/datas/artifacts/vocab.pkl','rb') as f:
                SAVE_TEXT = pickle.load(f)
            
            okt=Okt() 

            ID = Field(sequential = False, use_vocab = False)
            TEXT = Field(sequential = True,
                            use_vocab = True,
                            tokenize = okt.morphs,
                            lower = True,
                            batch_first = True,
                            fix_length = 20)
            
            fields = {"id": ID, "msg_body":TEXT}

            data = DataFrameDataset(self._msg_df, fields)

            TEXT.build_vocab(data)


    #         indexed = [TEXT.vocab.stoi[i] for i in TEXT.vocab.stoi.keys()]
            index = []
            
            for key, value in TEXT.vocab.stoi.items():
                if key in SAVE_TEXT:
                    index.append(SAVE_TEXT[key])
                else:
                    index.append(0)
                    
            tensor = torch.LongTensor(index).to(device)
            tensor = tensor.unsqueeze(0)
    #         print(TEXT.vocab.stoi)
    #         print(SAVE_TEXT)
    #         print("vocab:")
    #         print(index)
    #         for i in TEXT.vocab.stoi.keys():
    #             print(i)
    #             print(TEXT.vocab.stoi[i])
            
            return tensor
        else:
            print("(광고) 문구가 들어가있어서 통과되었음")
            return None

        

class MsgTrainModel:
    def __init__(self, repo_path:str, f_service_name:str, batch_size, device) -> None:
        self._repo_path = repo_path
        self._feature_service_name = f_service_name
        self._device = device
        self._batch_size = batch_size
        
        
    def get_training_data(self):
        orders = pd.read_csv("y_data.csv")
        orders.drop(['Unnamed: 0'],axis=1,inplace=True)
        
        for i, items in orders.iterrows():
            new_time = datetime.fromtimestamp(items['event_timestamp']/1000)#그냥 초가 아니라 밀리초로 되어있어서 1000을 나누어야한다
            orders.loc[i,'event_timestamp'] = new_time

        orders["event_timestamp"] = pd.to_datetime(orders["event_timestamp"])
        

        store = feast.FeatureStore(repo_path=self._repo_path)
        feature_service = store.get_feature_service(self._feature_service_name)
        
        training_df = store.get_historical_features(
            entity_df = orders,
            features = feature_service
        ).to_df()

        
        #category가 2인 열들 삭제
        idx = training_df[training_df['category']==2].index
        train_data_result = training_df.drop(idx)
        train_data_result.drop(['event_timestamp'],axis=1,inplace=True)
            
        
        
        okt=Okt() 

        #필드 정의
        ID = Field(sequential = False, use_vocab = False)
        TEXT = Field(sequential = True,
                          use_vocab = True,
                          tokenize = okt.morphs,
                          lower = True,
                          batch_first = True,
                          fix_length = 20)
        LABEL = LabelField()

        #데이터를 불러와서 데이터셋의 형식으로 바꿔주고, 그와 동시에 토큰화를 수행
        fields = {"id": ID, "msg_body":TEXT, "category":LABEL}

        train_df, valid_df = Split_DataFrame(train_data_result, 0.8, 200)

        train_data = DataFrameDataset(train_df, fields)
        test_data = DataFrameDataset(valid_df, fields)

        SEED =1234

        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



        print('훈련 샘플의 갯수: {}'.format(len(train_data)))
        print('테스트 샘플의 개수: {}'.format(len(test_data)))

        print(vars(train_data[0])) #확인용

        #단어 집합 만들기
        # min_freq: 단어 집합에 추가 시 단어의 최소 등장 빈도 조건을 추가
        # max_size: 단어 집합의 최대 크기를 지정
        TEXT.build_vocab(train_data, min_freq=2, max_size=10000)
        LABEL.build_vocab(train_data)

        #단어집합 저장
        with open('artifacts/vocab.pkl', 'wb') as f:
            pickle.dump(TEXT.vocab.stoi,f, protocol=pickle.HIGHEST_PROTOCOL)

        print('단어 집합의 크기: {}'.format(len(TEXT.vocab)))
        print('라벨의 갯수: {}'.format(len(LABEL.vocab)))

        print(TEXT.vocab.freqs.most_common(20))
        print(TEXT.vocab.stoi)
        print(LABEL.vocab.stoi)
        
        train_iterator, test_iterator = BucketIterator.splits((train_data, test_data),batch_size = self._batch_size, shuffle=True,sort=False, device = self._device)
        
        
        return train_iterator, test_iterator, len(TEXT.vocab)
    
    def train_model(self,model,optimizer,train_iter) -> None:
        model.train()
        for b, batch in enumerate(train_iter):
            x, y = batch.msg_body.to(self._device), batch.category.to(self._device)
    #         y.data.sub_(1)  # 레이블 값을 0과 1로 변환
            optimizer.zero_grad()

            logit = model(x)
            loss = F.cross_entropy(logit, y)
            loss.backward()
            optimizer.step()
            
    def evaluate_model(self,model, val_iter):
        """evaluate model"""
        model.eval()
        corrects, total_loss = 0, 0
        for batch in val_iter:
            x, y = batch.msg_body.to(self._device), batch.category.to(self._device)
    #         y.data.sub_(1) # 레이블 값을 0과 1로 변환
            logit = model(x)
            loss = F.cross_entropy(logit, y, reduction='sum')
            total_loss += loss.item()
            corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
        size = len(val_iter.dataset)
        avg_loss = total_loss / size
        avg_accuracy = 100.0 * corrects / size
        return avg_loss, avg_accuracy



    

