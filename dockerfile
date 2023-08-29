FROM nvcr.io/nvidia/tensorflow:21.12-tf1-py3

RUN pip install --upgrade pip

RUN pip3 install torch torchvision torchaudio
RUN pip3 install opencv-python