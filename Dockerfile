FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY model.h5 server.py /app
COPY templates /app/templates/

CMD ["python", "server.py"]
