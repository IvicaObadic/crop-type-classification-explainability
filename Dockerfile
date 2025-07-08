FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# MAINTAINER Ivica Obadic


RUN chmod -R 777 /tmp


RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    jq \
    git \
    iputils-ping \
    libcurl4 \
    # libicu60 \
    libunwind8 \
    netcat \
    libssl1.0 
  
#RUN pip install --upgrade pip

# Install required packages
RUN pip install torchsummary
RUN pip install torchtext
RUN pip install fastprogress
RUN pip install jupyter
RUN pip install pandas
RUN pip install matplotlib
RUN pip install rasterio
RUN pip install tqdm
RUN pip install tensorboard
RUN pip install scikit-learn
RUN pip install glob2
RUN pip install scikit-image




##Installing cv2
RUN pip install opencv-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

##Compilation libraries
RUN pip install gcc7
RUN apt-get update && apt-get install build-essential -y

##Other utlities
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN  apt-get update \
  && apt-get install unzip



RUN pip install cmake
# RUN pip install scikit-learn==0.22.2


WORKDIR /home/luca
COPY . /home/luca

CMD bash

EXPOSE 8888