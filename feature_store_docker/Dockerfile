FROM python:3.9
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre default-jdk

WORKDIR /workspace
RUN mkdir data
COPY . /workspace/code/datas

EXPOSE 21571
# RUN python get_data.py
# CMD ["feast","apply"]