FROM python:3.9
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre default-jdk

#WORKDIR 을 docker-compose에서 shared-volume의 경로와 같이 해줘야 제대로 생성된다.
WORKDIR /code/datas 
COPY . /code/datas

EXPOSE 21571
# RUN python get_data.py
# CMD ["feast","apply"]