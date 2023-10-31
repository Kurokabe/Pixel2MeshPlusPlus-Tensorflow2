FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN pip install tflearn==0.3.2
RUN pip install opencv-python
RUN pip install pyyaml
RUN pip install scipy