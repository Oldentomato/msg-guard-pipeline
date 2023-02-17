import torch
import torchtext
from torchtext.legacy.data import TabularDataset,Field, Dataset, Example,LabelField
import random
from konlpy.tag import Okt 
import pandas as pd


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



def Create_DataSet(getdatas):
    #Mecab을 토크나이저로 사용
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
    
    #데이터를 불러와서 데이터셋의 형식으로 바꿔주고, 그와 동시에 토큰화를 수행

    fields = {"count": None, "id": ID, "msg_body":TEXT, "category":LABEL}

    train_ds = DataFrameDataset(train_df, fields)
    valid_ds = DataFrameDataset(valid_df, fields)

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

    return train_data, test_data, len(TEXT.vocab)



    

