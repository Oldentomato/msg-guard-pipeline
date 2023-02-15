from torchtext import data # torchtext.data 임포트
from konlpy.tag import Mecab
from torchtext.data import TabularDataset

def Create_DataSet(getdatas):
    #Mecab을 토크나이저로 사용
    tokenizer = Mecab()

    #필드 정의
    ID = data.Field(sequential = False, use_vocab = False)
    TEXT = data.Field(sequential = True,
                      use_vocab = True,
                      tokenize = tokenizer.morphs, #토크나이저로는 Mecab 사용
                      lower = True,
                      batch_first = True,
                      fix_length = 20)
    LABEL = data.Field(sequential = False,
                       use_vocab = False,
                       is_target = True)
    
    #데이터를 불러와서 데이터셋의 형식으로 바꿔주고, 그와 동시에 토큰화를 수행
    train_data, test_data = TabularDataset.splits(
        path='.', train = getdatas.train, test = getdatas.test, format='tsv', fields=[('id',ID),('text',TEXT),('label',LABEL)], skip_header=True
    )

    print('훈련 샘플의 갯수: {}'.format(len(train_data)))
    print('테스트 샘플의 개수: {}'.fomat(len(test_data)))

    print(vars(train_data[0])) #확인용

    #단어 집합 만들기
    # min_freq: 단어 집합에 추가 시 단어의 최소 등장 빈도 조건을 추가
    # max_size: 단어 집합의 최대 크기를 지정
    TEXT.build_vacab(train_data, min_freq=10, max_size=10000)

    print('단어 집합의 크기: {}'.format(len(TEXT.vocab)))
    print(TEXT.vocab.stoi)

    #토치 텍스트의 데이터로더 만들기
    

