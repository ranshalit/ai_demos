### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)
6. [Quantize](#quantize)
7. [Acknowledgement](#acknowledgement)

### Installation

1. Environment requirement
    - anaconda3
    - tensorflow 2.3
    - cython, opencv, tqdm etc. (refer to [requirements](requirements.txt) for more details)

2. Installation
   ```shell
   conda create -n yolov3-tf2 python=3.6
   source activate yolov3-tf2
   # conda activate yolov3-tf2
   pip install cython
   pip install --user -r requirements.txt
   ```

### Preparation

1. Dataset description

The dataset for evaluation is MSCOCO val2017 set which contains 5000 images.

2. Download and prepare the dataset

Run the script `prepare_data.sh` to download and prepare the dataset.
   ```shell
   bash code/test/download_data.sh
   bash code/test/convert_data.sh
   ```
Dataset diretory structure: 
   ```shell
   # val2017 and annotations are unpacked from the downloaded data
   + data
     + val2017
       + 000000000139.jpg
       + 000000000285.jpg
       + ...
     + annotations
       + instances_train2017.json
       + instances_val2017.json
       + ...
     + val2017.txt
   ```

3. Download the official darknet weights and convert to tensorflow weights of .h5 format
   ```
   cd code/test
   wget -O ../../float/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
   python tools/model_converter/convert.py cfg/yolov3.cfg ../../float/yolov3.weights ../../float/yolov3.h5
   ```

### Train/Eval

1. Evaluation
    Configure the model path and data path in [code/test/run_eval.sh](code/test/run_eval.sh)
    ```shell
    # run the script under 'code/test'
    bash run_eval.sh
    ```

### Performance

|Model|Input size|FLOPs|Params|Dataset|mAP(IoU=0.50:0.95)|
|:-:|:-:|:-:|:-:|:-:|:-:|
|yolov3_coco|416x416|65.9G|62.0M|COCO 2017val|37.7%|


### Model_info

1. Data preprocess
  ```
  data channel order: RGB(0~255)
  resize: keep aspect ratio of the raw image and resize it to make the length of the longer side equal to 416
  padding: pad along the short side with pixel value 128 to generate the input image with size = 416 x 416
  input = input / 255
  ``` 

2. Node information
  ```
  input node: 'image_input:0'
  output nodes: 'conv2d_58:0', 'conv2d_66:0', 'conv2d_74:0'
  ```

### Quantize
1. Quantize tool installation
  See [vai_q_tensorflow](https://github.com/Xilinx/Vitis-AI/tree/master/Vitis-AI-Quantizer/vai_q_tensorflow)

2. Quantize workspace
  See [quantize](./code/quantize/)

3. Quantized model performance
|Model|Input size|Dataset|mAP(IoU=0.50:0.95)|
|:-:|:-:|:-:|:-:|
|yolov3|416x416|COCO 2017val|33.1%| 

### Acknowledgement

[keras-YOLOv3-model-set](https://github.com/david8862/keras-YOLOv3-model-set.git)
