FROM python:3.9
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre default-jdk
WORKDIR /code/datas
COPY . /manager
RUN mkdir /manager/artifacts

EXPOSE 80
# CMD ["uvicorn","app.api:app","--host","0.0.0.0","--port","80"]
# CMD ["python","train.py"]