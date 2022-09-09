FROM python:3.8

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENV PYTHONPATH /app
ENV HEIGHT=10
ENV WIDTH=10
ENV EPOCHS=100

CMD python -m src.train --height ${HEIGHT} --width ${WIDTH} --epochs ${EPOCHS}
