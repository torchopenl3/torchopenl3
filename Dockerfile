FROM ubuntu:20.04
ENV LANG C.UTF-8

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Add non root user
RUN useradd -ms /bin/bash openl3 && echo "openl3:openl3" | chpasswd && adduser openl3 sudo

WORKDIR /home/openl3

RUN apt-get update
RUN apt-get install -y software-properties-common apt-file
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.6 curl
RUN apt-get remove -y python3.8-minimal
RUN update-alternatives --install /usr/local/bin/python python /usr/bin/python3.6 3
RUN update-alternatives --install /usr/local/bin/python3 python /usr/bin/python3.6 3
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.6 get-pip.py && rm get-pip.py
RUN apt-get install -y git vim sudo
RUN apt-get install -y libsndfile-dev

RUN git clone https://github.com/turian/torchopenl3
RUN cd torchopenl3 && pip3.6 install -e ".[dev]"
RUN pip3 uninstall torchopenl3
RUN rm -Rf torchopenl3

RUN apt-get install -y screen
RUN apt-get autoremove -y
RUN rm -Rf /root/.cache

USER openl3
