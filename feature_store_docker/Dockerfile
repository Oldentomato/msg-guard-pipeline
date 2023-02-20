FROM python:3.9
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN sudo apt-get install openjdk-8-jdk

WORKDIR /feature_store
COPY . /feature_store
ADD ./example_file.py /feature_repo
RUN mkdir -p /feature_store/data
ADD ./feature_store.yaml /feature_repo

EXPOSE 21571
CMD ["python","get_data.py"]