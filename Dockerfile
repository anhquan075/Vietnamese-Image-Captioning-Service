FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install python3-pip -y && pip3 install -U pip && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

ENV  PYTHONIOENCODING=utf8

COPY requirements.txt prepare_service.sh /app/

RUN pip3 install -r requirements.txt

RUN bash prepare_service.sh

COPY . /app/ 

EXPOSE 5000

CMD ["python3", "main.py"]