## Reference

Vitis-ai quick start:  
[https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html](https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html)  
[https://github.com/Xilinx/Vitis-AI](https://github.com/Xilinx/Vitis-AI)  
Usb3 install:  
[https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18841729/DWC3+Xilinx+Linux+USB+driver](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18841729/DWC3+Xilinx+Linux+USB+driver)

## 

## General Setup

* Zcu102 board  
* Camera usb3 \- fix switches as described  
  [https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/1586626609/USB+Boot+example+using+ZCU102+Host+and+ZCU102+Device](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/1586626609/USB+Boot+example+using+ZCU102+Host+and+ZCU102+Device)  
  3 switches in total \- we had to change the defaults  
  * J7 – 1-2 Open   
  * J113 \- 1-2 Close  
  * J110 \- 1-2 Close  
* Once we connect a simple camera we can see in dmesg that it is detected  
* SD card with image for zcu102 see above link  
* Serial to usb  
* Ethernet connection

Note: we have 2 zcu102 with rev 1.1, yet on running examples on one of them it fails on failed fingerprints, i.e. 2 cores with correct finger print but the 3rd one has a fault finger print  
TODO: should try to run \_optimize script see vitis-ai q\&a

various examples:  
1\. stability of image  
2\. detect of target (person, car, etc)  
3\. Tracking of objects

## Simple utilities to check format and resolution

Check resolution:  
*ffprobe video/adas.webm*  
convert video to tiff in 1920x1080 resolution:  
*ffmpeg \-i \*.webm \-vf scale=1920:1080 output\_%04d.tiff*

modify readfile in main.cc of adas\_detection

## Prepare docker

There are some vitis-ai docker in hub  
[https://hub.docker.com/search?q=xilinx+vitis](https://hub.docker.com/search?q=xilinx+vitis)  
Useful:  
[https://hub.docker.com/r/xilinx/vitis-ai-pytorch-cpu](https://hub.docker.com/r/xilinx/vitis-ai-pytorch-cpu)  
[https://hub.docker.com/r/xilinx/vitis-ai-tensorflow2-cpu](https://hub.docker.com/r/xilinx/vitis-ai-tensorflow2-cpu)  
Note: there is also xilinx/vitis-ai-cpu but it seemed older so I did not use it.

But I wanted gpu tf2 and did not find it so built it myself here:  
cd\~/projects/vitis-ai/Vitis-AI/docker  
./docker\_build.sh \-f tf2 \-t gpu

Enter docker using:  
sudo ./docker\_run.sh xilinx/vitis-ai-tensorflow2-gpu:latest  
And then inside docker:  
conda activate vitis-ai-tensorflow2

## Build application for AI model

2 options:

1. Build on host using cross compiler ([https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html](https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html))  
   \[Host\] $ cd Vitis-AI/board\_setup/mpsoc  
   Only for the first time:  
   \[Host\] $ sudo chmod u+r+x host\_cross\_compiler\_setup.sh  
   \[Host\] $ ./host\_cross\_compiler\_setup.sh 

When the installation is complete, follow the prompts and execute the following command:

\[Host\] $ source \~/petalinux\_sdk\_2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux  
cd \~/projects/vitis-ai/Vitis-AI/examples/vai\_runtime/adas\_detection  
bash \+x build.sh

2. Run docker source cross-compiler and build  
   cd \~/projects/vitis-ai/Vitis-AI/examples/vai\_runtime/adas\_detection  
   bash \+x build.sh  
   

SD IMAGE backup  
[https://drive.google.com/file/d/174lMQa\_VEUNpPDdsyLyodPiHVbvvMiBP/view?usp=sharing](https://drive.google.com/file/d/174lMQa_VEUNpPDdsyLyodPiHVbvvMiBP/view?usp=sharing)  
[https://drive.google.com/file/d/1v2iE4ZZOMogsV0mNDJXu6V1RWs8Xniro/view?usp=sharing](https://drive.google.com/file/d/1v2iE4ZZOMogsV0mNDJXu6V1RWs8Xniro/view?usp=sharing)  
Vitis-AI code change backup  
[https://github.com/ranshalit/ai\_demos](https://github.com/ranshalit/ai_demos)

Run Application example

\[HOST\]   
Sudo ifconfig \<\> 192.168.1.101  
scp adas\_detection root@192.168.1.100  
scp tiff folder to /tmp in target  
ssh \-X root@192.168.1.100

Run any example according to table in:  
[https://github.com/Xilinx/Vitis-AI/blob/master/examples/vai\_runtime/README.md](https://github.com/Xilinx/Vitis-AI/blob/master/examples/vai_runtime/README.md)  
E.g.  
./adas\_detection video/adas.webm /usr/share/vitis\_ai\_library/models/yolov3\_adas\_pruned\_0\_9/yolov3\_adas\_pruned\_0\_9.xmodel

Running our own built “adas\_demo”:  
in target:  
   "Usage of ADAS detection: "   
  " \<video\> \<model\_file\> \<input type\> \<model type\>"  
  "input type: (0-video file (.webm), 1-folder of tiffs, 2-camera (dev/video0)"  
  "model type: 0-adas, 1-yolo3 2-custom model"

E.g.  
Running on 80 class darknet yolo3 model:  
\~/adas\_detection /dev/video0 \~/yolov3\_tf2.xmodel 0 1  
Running on my own yolo3 trained model (20 classes)  
\~/adas\_detection /dev/video0 \~/test.xmodel 0 2

Download model

# Evaluate float and quantize model (tf2)

### Download model code

cd \~/projects/vitis-ai/Vitis-AI/model\_zoo/  
python 3 [download.py](http://download.py)  
choose tf2 yolov3\_coco\_416\_416\_65.9G\_3.0  
extract tar and follow the [README.md](http://README.md)  \!

If you use docker then no need to do these install from readme:  
(Environment requirement

* anaconda3  
  * tensorflow 2.3  
  * cython, opencv, tqdm etc. (refer to [requirements](https://file+.vscode-resource.vscode-cdn.net/home/ranshal/projects/vitis-ai/Vitis-AI/model_zoo/tf2_yolov3_coco_416_416_65.9G_3.0/requirements.txt) for more details)

Installation  
conda create \-n yolov3-tf2 python=3.6  
source activate yolov3-tf2  
*\# conda activate yolov3-tf2*  
pip install python  
pip install \--user \-r requirements.txt

### Prepare dataset

**bash code/test/download\_data.sh**  
**bash code/test/convert\_data.sh**

Images will be extracted to  
/home/ranshal/projects/vitis-ai/Vitis-AI/model\_zoo/tf2\_yolov3\_coco\_416\_416\_65.9G\_3.0/data/val2017/000000500565.jpg  
….  
And annotation into:  
/home/ranshal/projects/vitis-ai/Vitis-AI/model\_zoo/tf2\_yolov3\_coco\_416\_416\_65.9G\_3.0/data/annotations/instances\_val2017.json  
…..

Dataset directory structure:

*\# val2017 and annotations are unpacked from the downloaded data*  
\+ data  
  \+ val2017  
    \+ 000000000139.jpg  
    \+ 000000000285.jpg  
    \+ ...  
  \+ annotations  
    \+ instances\_train2017.json  
    \+ instances\_val2017.json  
    \+ ...  
  \+ val2017.txt  \<- I think this is the result of prepare\_data.sh

Download the official darknet weights and convert to tensorflow weights of .h5 format  
**cd code/test**  
**wget \-O ../../float/yolov3.weights https://github.com/AlexeyAB/darknet/releases/download/darknet\_yolo\_v3\_optimal/yolov3.weight**  
**python tools/model\_converter/convert.py cfg/yolov3.cfg ../../float/yolov3.weights ../../float/yolov3.h5**  
**mkdir custom\_weight**  
**wget https://github.com/AlexeyAB/darknet/releases/download/darknet\_yolo\_v3\_optimal/yolov3.weights**

### Evaluate accuracy of float model (tf2)

Evaluation Configure the model path and data path in [code/test/run\_eval.sh](https://file+.vscode-resource.vscode-cdn.net/home/ranshal/projects/vitis-ai/Vitis-AI/model_zoo/tf2_yolov3_coco_416_416_65.9G_3.0/code/test/run_eval.sh)  
*\# run the script under 'code/test'*  
**bash run\_eval.sh**

### About model:

1. Data preprocess

data channel order: RGB(0\~255)  
resize: keep aspect ratio of the raw image and resize it to make the length of the longer side equal to 416  
padding: pad along the short side with pixel value 128 to generate the input image with size \= 416 x 416  
input \= input / 255

2. Node information

input node: 'image\_input:0'  
output nodes: 'conv2d\_58:0', 'conv2d\_66:0', 'conv2d\_74:0'

	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012\_img\_train.tar \--no-check-certificate  
	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012\_img\_val.tar \--no-check-certificate

cp code/gen\_data/val.txt data/validation/val.txt  
cp code/gen\_data/val\_filenames.txt data/validation/val\_filenames.txt  
cp code/gen\_data/ILSVRC2012\_validation\_ground\_truth.txt data/validation/ILSVRC2012\_validation\_ground\_truth.txt

Evaluate accuracy of quant model (tf2)  
To evaluate the quantized model after you have generated it (saved as ./quantized/quantized.h5), you can use the same eval.py script but with the quantized model path and the \--quant flag to indicate it's a quantized model.  
Here's how to do it:

1st option is to run    
**/code/test/run\_eval\_qual.sh**  
But you might need to edit that file (see the links to files at the start)

The 2nd option is to use these 2 calls:

1. **python eval.py   \--model\_path=./quantized/quantized.h5 \--anchors\_path=configs/yolo3\_anchors.txt   \--classes\_path=configs/coco\_classes.txt   \--annotation\_file=../../data/val2017.txt   \--model\_input\_shape=416x416  \--quant   \--eval\_only**  
2. **python tools/evaluation/pycoco\_eval.py   \--result\_txt result/detection\_result.txt \--coco\_annotation\_json ../../data/annotations/instances\_val2017.json   \--coco\_result\_json result/detection\_result\_iou0.45.json**

## Compile(tf2)

see https://www.hackster.io/LogicTronix/zcu102-vitis-dpu-trd-vitis-ai-3-0-c51609

   
**vai\_c\_tensorflow2 \\**  
  **\--model quantized/quantized.h5 \\**  
  **\--arch /opt/vitis\_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json \\**  
  **\--output\_dir compiled\_model \\**  
  **\--net\_name yolov3\_tf2 \\**  
  **\--options "{'input\_shape':'1,416,416,3'}"**  
   
Note: if during running example it complaint about wrong fingerprint, then try to modify arch.json in this command  
 

## Quantize (tf2)

As a preparation might need to create val2017.txt, a file which has links to dataset and provide annotation to each of the image, i.e. this format   
ranshal@D-P-RANSHAL-L-RF-L-RF:\~/projects/vitis-ai/Vitis-AI/model\_zoo/tf2\_yolov3\_coco\_416\_416\_65.9G\_3.0$ head data/val2017.txt   
/workspace/model\_zoo/tf2\_yolov3\_coco\_416\_416\_65.9G\_3.0/data/val2017/000000289343.jpg 473,395,511,423,16 204,235,264,412,0 0,499,339,605,13 204,304,256,456,1  
/workspace/model\_zoo/tf2\_yolov3\_coco\_416\_416\_65.9G\_3.0/data/val2017/000000061471.jpg 272,200,423,479,16 181,86,208,159,39 174,0,435,220,61

But no need to create it yourself\! Just run the required preparation script  
It depends on annotation format for example for coco use:

**python ./tf2\_yolov3\_coco\_416\_416\_65.9G\_3.0/code/test/tools/dataset\_converter/coco\_annotation\_val.py**  
this will generate a file like  
./../data/val2017.txt \\

**Note:**

| `coco_annotation.py` | Converts the training set (e.g., `train2017`) COCO annotations into YOLO/other formats |
| :---- | :---- |

| `coco_annotation_val.py` | Converts the validation set (e.g., `val2017`) COCO annotations into YOLO/other formats |
| :---- | :---- |

**python ./tf2\_yolov3\_coco\_416\_416\_65.9G\_3.0/code/test/tools/dataset\_converter/coco\_annotation.py**  
./../data/train2017.txt \\

**python ./tf2\_yolov3\_coco\_416\_416\_65.9G\_3.0/code/test/tools/dataset\_converter/coco\_annotation\_val.py**  
this will generate a file like  
./../data/val2017.txt \\

 **python eval.py \--model\_path=../../float/yolov3.h5  \--anchors\_path=configs/yolo3\_anchors.txt \--classes\_path=configs/coco\_classes.txt \--annotation\_file=../../data/val2017.txt \--model\_input\_shape=416x416 \--quant**

The quantized model will be saved here:

./quantized/quantized.h5

    

# Training (darknet)

Note: darknet is alternative to tf2, and originally yolo3 was created in darknet

Do it all from host (no docker):

Doing it with AlexeyAB’s repo and VOC sample:

1. **Clone AlexeyAB darknet repo:**

**git clone https://github.com/AlexeyAB/darknet.git**  
cd darknet

2. **Compile darknet:**

**sed \-i 's/GPU=0/GPU=1/' Makefile**  
**sed \-i 's/CUDNN=0/CUDNN=1/' Makefile**  
**sed \-i 's/OPENCV=0/OPENCV=1/' Makefile**  
**make \-j$(nproc)**  
Note: Actually I worked with GPU=1 and CUDNN=0, OPENCV=1, maybe that’s why it was quite slow (18 hours for 50000 iterations)

3. **Download VOC dataset and config files:**

**mkdir data/voc**  
**cd data/voc**  
**wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval\_11-May-2012.tar**   
**tar xf VOCtrainval\_11-May-2012.tar**  
**cd ../..**  
**python3 voc\_label.py**   
**cd tools/dataset\_converter/**  
**python voc\_annotation.py    \--dataset\_path /workspace/darknet/data/voc/VOCdevkit  \--year 2012 \--set train    \--output\_path  . \--classes\_path /workspace/darknet/data/voc.names**

this will generate a file like:  
**2012\_train.txt,** we then copy to darknet main folder

4. **Download pretrained weights for darknet53 backbone:**

[https://sourceforge.net/projects/yolov3.mirror/files/v8/darknet53.conv.74/download](https://sourceforge.net/projects/yolov3.mirror/files/v8/darknet53.conv.74/download)  
cp darknet53.conv.74  backup/

5. **Start training on VOC (20 classes):**

vi cfg/voc.data  
*classes= 20*  
*train  \= /home/ranshal/projects/vitis-ai/Vitis-AI/darknet/data/voc/2012\_train.txt*  
*valid  \= /home/ranshal/projects/vitis-ai/Vitis-AI/darknet/data/voc/2012\_val.txt*  
*names \= data/voc.names*  
*backup \= /home/ranshal/projects/vitis-ai/Vitis-AI/darknet/backup*

vi cfg/yolov3-voc.cfg

*\[net\]*  
*\# Testing*  
*batch=8 ←-*  
*subdivisions=2 ←-*  
*\# Training*  
*\# batch=64*  
*\# subdivisions=16*  
*width=416*  
*height=416*  
*channels=3*  
*momentum=0.9*  
*decay=0.0005*  
*angle=0*

Also set random=0 in all relevant places, otherwise it may create random size of images

./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg backup/darknet53.conv.74 \-dont\_show 

6. **Convert weight(darknet) to h5(tensorflow)**  
   cd workspace/model\_zoo/tf2\_yolov3\_coco\_416\_416\_65.9G\_3.0/code/test/tools/model\_converter  
   **python3 convert.py    /workspace/darknet/cfg/yolov3-voc.cfg   /workspace/darknet/yolov3-voc\_last.weights    /workspace/darknet/yolov3\_converted.h5**

7. **Quantize h5**

**python eval.py \--model\_path=/workspace/darknet/yolov3\_converted.h5  \--anchors\_path=/workspace/darknet/yolov3\_converted\_anchors.txt \--classes\_path=/workspace/darknet/data/voc.names \--annotation\_file=/workspace/darknet/2012\_train.txt \--model\_input\_shape=416x416 \--quant**

8. **Compile the quantized model h5 into xmodel**

**vai\_c\_tensorflow2   \--model /workspace/darknet/quantized.h5   \--arch /opt/vitis\_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json   \--output\_dir compiled\_model   \--net\_name yolov3\_tf2   \--options "{'input\_shape':'1,416,416,3'}"**

Hooray\! We got a model ./compiled\_model/yolov3\_tf2.xmodel

## Useful utilities

xdputil xmodel \-l compiled\_model/yolov3\_tf2.xmodel  
Now we can use this name in our application \!

## Build kernel and petalinux

Build using petalinux [https://github.com/Xilinx/Vitis-AI/blob/v3.0/src/vai\_petalinux\_recipes/recipes-vai-kernel/linux-xlnx/bsp.cfg](https://github.com/Xilinx/Vitis-AI/blob/v3.0/src/vai_petalinux_recipes/recipes-vai-kernel/linux-xlnx/bsp.cfg)   
Added manually  
Also added CONFIG\_DMA\_PROXY

## Application

I used adas\_detection example and modified it:

* Added camera input  
* Added support for different model: yolo3 (tf2 \- 80 class), yolo3(darknet 20 class)  
* Added input options: folder of tiffs

   "Usage of ADAS detection: "   
  " \<video\> \<model\_file\> \<input type\> \<model type\>"  
  "input type: (0-video file (.webm), 1-folder of tiffs, 2-camera (dev/video0)"  
  "model type: 0-adas, 1-yolo3 2-custome model"

## Appendix

## Resnet50 model evaluation

\#\# ✅ Step-by-Step Instructions

\#\#\# 📁 1\. Move the file to your project directory

Assuming your working directory is:

\`\`\`  
/workspace/model\_zoo/tf\_resnetv1\_50\_imagenet\_224\_224\_6.97G\_3.0/  
\`\`\`

Move the tar file there (if not already there):

mv \~/Downloads/ILSVRC2012\_img\_val.tar ./data/

After that, it should exist as:

./data/ILSVRC2012\_img\_val.tar

\#\#\# 📦 2\. Extract the images to the \`validation/\` folder

\`\`\`bash  
mkdir \-p ./data/validation  
tar \-xf ./data/ILSVRC2012\_img\_val.tar \-C ./data/validation/  
\`\`\`

This creates:  
./data/validation/ILSVRC2012\_val\_00000001.JPEG  
./data/validation/ILSVRC2012\_val\_00000002.JPEG

 You should see around \*\*50,000 JPEG files\*\*.

\#\#\# 🧪 3\. Verify the extraction

Run:

\`\`\`bash  
ls ./data/validation | wc \-l  
\`\`\`

Should output:

\`\`\`  
50000  
\`\`\`

\---

\#\#\# 🧾 4\. Run \`get\_dataset.sh\` to preprocess the dataset

Now go to the script folder and run:

\`\`\`bash  
cd code/gen\_data  
./get\_dataset.sh  
\`\`\`

This script will:

\* Take the raw images from \`../../data/validation/\`  
\* Generate:

  \* \`../../data/Imagenet/val\_dataset/\` → resized & cropped images  
  \* \`../../data/Imagenet/val.txt\` → labels

\---

\#\#\# ✅ 5\. Validate Output

Check a few files:

\`\`\`bash  
ls ../../data/Imagenet/val\_dataset | head  
head ../../data/Imagenet/val.txt  
\`\`\`

Also count:

\`\`\`bash  
ls ../../data/Imagenet/val\_dataset | wc \-l  
wc \-l ../../data/Imagenet/val.txt  
\`\`\`

Expect both to be \\\~50,000.

\---

You're now ready to run:

\`\`\`bash  
cd ../test  
bash run\_eval\_float\_pb.sh  
\`\`\`

vai\_c\_tensorflow \--arch ./arch\_zcu102\_2xb512\_march2023.json \-f quantized/quantized\_baseline\_6.96B\_919.pb \--output\_dir compile\_result\_zcu102 \-n tf\_resnet50