FROM python:3.9
COPY requirements.txt requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre default-jdk
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
volume ["/app/datas"]
WORKDIR /app/datas
COPY . /app/datas
RUN mkdir /app/datas/artifacts
RUN python get_data.py && \
    feast apply && \
    python train.py

EXPOSE 80
CMD ["uvicorn","app.api:app","--host","0.0.0.0","--port","80"]
