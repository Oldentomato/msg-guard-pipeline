FROM python:3.9
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
WORKDIR /feature_store
COPY . /feature_store
EXPOSE 21571
CMD ["python","get_data.py"]