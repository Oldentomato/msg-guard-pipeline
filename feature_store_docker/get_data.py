#spark 세션 만들기
from pyspark.sql import SparkSession
import pandas as pd
import re

def GetDataFromMongoSpark():
    spark = SparkSession \
    .builder \
    .appName("msg_guard") \
    .master("local") \
    .config("spark.mongodb.input.uri","mongodb+srv://Oldentomato:jowoosung123@examplecluster.g7o5t.mongodb.net/SecondDatabase?retryWrites=true&w=majority") \
    .config("spark.mongodb.output.uri","mongodb+srv://Oldentomato:jowoosung123@examplecluster.g7o5t.mongodb.net/SecondDatabase?retryWrites=true&w=majority") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .getOrCreate()


    df = spark.read.format('com.mongodb.spark.sql.DefaultSource')\
    .option("uri","mongodb+srv://Oldentomato:jowoosung123@examplecluster.g7o5t.mongodb.net/SecondDatabase?retryWrites=true&w=majority")\
    .option("database","Msg_Database")\
    .option("collection","MSG_RowData")\
    .load()

    df.printSchema()#확인용

    return df.toPandas()

def SetCleanData(pddf):
    final_msgs = []
    for count in range(0,len(pddf['msg_data'])):
        for msg in pddf['msg_data'][count]:
            if msg['msg_body'].find('(광고)') == -1:
                final_msgs.append({'id':msg['msg_id'],'msg_body':msg['msg_body']})

    msg_df = pd.DataFrame(final_msgs)            

    print(msg_df)#확인용



    #[Web발신] 글자 삭제 및 한글만 남기기, timestamp에서 datetime으로 변환
    for i,items in msg_df.iterrows():
        new_str = items['msg_body'].replace('[Web발신]','')
        hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
        result = hangul.sub('',new_str)
        msg_df.loc[i,'msg_body'] = result
        
    print(msg_df)

    #feature store를 위한 parquet 파일 저장
    msg_df.to_parquet('/workspace/feature_store/data/msg_data.parquet')


    #url 감지(이건 나중에)
    # str1 = pddf['msg_data'][0][5]['msg_body']
    # m = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$\-@\.&+:/?=]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str1)

if __name__ == '__main__':
    get_mongo_data = GetDataFromMongoSpark()
    SetCleanData(get_mongo_data)
