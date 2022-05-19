#!/bin/bash
echo "installing Module"
apt-get install protobuf-compiler
git clone https://github.com/tensorflow/models .
pip install opencv-python tensorflow numpy
cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . 