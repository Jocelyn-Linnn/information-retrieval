FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install flask

ENV FLASK_APP=server.py
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["flask", "run"]
