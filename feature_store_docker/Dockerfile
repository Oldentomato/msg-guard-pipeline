FROM python:3.9
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre default-jdk

WORKDIR /feature_store
RUN mkdir data
COPY . /feature_store

EXPOSE 21571
RUN python get_data.py
CMD ["feast","apply"]