FROM ubuntu:20.04 ENV LANG C.UTF-8

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Add non root user
# FIXME
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

#RUN apt-get install -y lsb-release wget software-properties-common
#RUN apt-get install -y git build-essential python3-pip git libsndfile-dev vorbis-tools lsb-release wget software-properties-common sudo less bc screen tmux unzip vim wget openssh-client libasound2-dev
# vorbis-tools
#RUN apt-get install -y lsb-release wget software-properties-common
#RUN apt-get install -y libsndfile-dev libhdf5-dev python3-h5py
#RUN apt-get install -y llvm-10*
#RUN ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config
#
## Everything from setup.py
##RUN pip3 install numpy scipy soundfile resampy torch "nnAudio" "pytest" "pytest-cov" "Cython >= 0.23.4" "openl3==0.3.1" "kapre==0.1.4" "h5py==2.10.0" "tensorflow<1.14" "requests" "tqdm" "pre-commit" "nbstripout==0.3.9" "black==20.8b1" "jupytext==v1.10.3"
##RUN pip3 install "numpy>=1.13.0" "scipy>=0.19.1" "soundfile" "resampy>=0.2.1,<0.3.0" "torch>=1.4.0" "nnAudio" "pytest" "pytest-cov" "Cython >= 0.23.4" "openl3==0.3.1" "kapre==0.1.4" "h5py==2.10.0" "tensorflow<1.14" "requests" "tqdm" "pre-commit" "nbstripout==0.3.9" "black==20.8b1" "jupytext==v1.10.3"
#
RUN git clone https://github.com/turian/torchopenl3
RUN cd torchopenl3 && pip3.6 install -e ".[dev]"
RUN pip3 uninstall torchopenl3
RUN rm -Rf torchopenl3

RUN apt-get autoremove -y

USER openl3
##RUN git clone https://github.com/turian/torchopenl3/
##RUN cd torchopenl3 && git checkout regression
#
##USER root
##RUN cd torchopenl3 && pip3 install -e "."
##RUN cd torchopenl3 && pip3 install -e ".[dev]"
