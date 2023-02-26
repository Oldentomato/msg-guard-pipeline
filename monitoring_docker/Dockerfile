FROM python:3.9
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
WORKDIR /workspace
COPY . /workspace

EXPOSE 80
# CMD ["uvicorn","app.api:app","--host","0.0.0.0","--port","80"]
# CMD ["python","train.py"]