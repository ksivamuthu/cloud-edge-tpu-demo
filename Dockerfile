FROM  python:3.7-buster

ENV UDEV=1

RUN wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_x86_64.whl
RUN pip3 install tflite_runtime-1.14.0-cp37-cp37m-linux_x86_64.whl

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update

RUN apt-get install libedgetpu1-std -y
RUN apt-get install edgetpu -y

COPY ./ /usr/src/app
WORKDIR /usr/src/app
RUN cd classification && bash install_requirements.sh

ENTRYPOINT [ "/bin/bash" ]