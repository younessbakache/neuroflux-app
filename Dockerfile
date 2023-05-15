FROM python:3.8

COPY  requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

RUN pip install protobuf==3.20.0

CMD ["python", "./main.py"]