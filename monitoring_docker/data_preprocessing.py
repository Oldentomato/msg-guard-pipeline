import torch
import torchtext
from torchtext.legacy.data import Field, Dataset, Example,LabelField
from konlpy.tag import Okt 
import pandas as pd
from feast import FeatureStore
import datetime


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
    
def __Split_DataFrame(df, frac, random_state):
    train = df.sample(frac=frac, random_state=random_state)
    test = df.drop(train.index)

    return train, test



def Create_DataSet():

    entity_df = pd.DataFrame.from_dict({
    "id": [],
    "event_timestamp": datetime(2022,10,5,12,50,4) #datetime 은 가장 마지막 시간보다 크게해야함
    })


    store = FeatureStore(repo_path="/workspace/feature_store/feature_repo/")


    # feature = store.get_online_features(
    #     features=[
    #         "msg_datas:msg_body",
    #         "msg_datas:category"
    #     ],
    #     entity_rows=[
    #         {"id": 3031}
    #     ]
    # ).to_dict()

    # pprint(feature)
    training_df = store.get_historical_features(
        entity_df=entity_df,  # 위에서 만든 데이터프레임을 넘겨준다.
        features = [
            'msg_datas:msg_body',
            'msg_datas:category',
        ],  # 불러올 feature를 적는다.
    ).to_df()

    # training_df.head()

    okt=Okt() 

    #필드 정의
    ID = Field(sequential = False, use_vocab = False)
    TEXT = Field(sequential = True,
                      use_vocab = True,
                      tokenize = okt.morphs, #토크나이저로는 Mecab 사용
                      lower = True,
                      batch_first = True,
                      fix_length = 20)
    
    LABEL = LabelField()

    SEED =1234
    
    #데이터를 불러와서 데이터셋의 형식으로 바꿔주고, 그와 동시에 토큰화를 수행

    fields = {"count": None, "id": ID, "msg_body":TEXT, "category":LABEL}

    train_df, valid_df = __Split_DataFrame(training_df, 0.8, 200)

    train_data = DataFrameDataset(train_df, fields)
    test_data = DataFrameDataset(valid_df, fields)



    # all_datas = TabularDataset(
    #     'result_data.csv', format='csv', fields=[("count",None),('id',ID),('msg_body',TEXT),('category',LABEL)], skip_header=True
    # )


    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # train_data, test_data = all_datas.split(split_ratio=0.8, stratified=False, strata_field = 'label', random_state = random.seed(SEED))

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

    return train_data, test_data, len(TEXT.vocab)



    

