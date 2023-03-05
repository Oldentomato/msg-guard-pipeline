FROM python:3.9
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre default-jdk
WORKDIR /workspace
COPY . /workspace
RUN mkdir /artifacts
RUN ["python","get_data.py"]
RUN ["feast","apply"]
RUN ["python","train.py"]

EXPOSE 80
CMD ["uvicorn","app.api:app","--host","0.0.0.0","--port","80"]
# CMD ["python","train.py"]