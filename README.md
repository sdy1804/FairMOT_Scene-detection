# FairMOT-Scene-detection
![Image](https://github.com/user-attachments/assets/6331f2fc-5f00-4dbe-b6aa-2ea8123537e7)

## Introduction
![Model architecture](./architecture.png)    
In swimming competition video, there are some scenes that repeatly occured when swimmers do competition.     
We classified some scenes - On-block, Diving, Swimming, Turning, Finish.    
Optimized hyper-parameters were different for each classes, so to utilize this we added classification head to FairMOT.    
After classification head classified scene classes, hyper-parameters are changed by each optimized values for classes.     
We increase swimmer tracking persistance and MOTA score maximum 5.2% than before not used classification head.    
You can find details in our [paper](http://ieiespc.org/ieiespc/XmlViewer/f431390).

## Installation
You can use these environments to implement. We used Python=3.8 and CUDA 11.2 in Ubuntu.    
```
conda create -n FairMOT python=3.8
conda activate FairMOT
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```
And please follow [FairMOT](https://github.com/ifzhang/FairMOT)'s installation.    

## Swimming scene detection dataset
We trained FairMOT classification head using this [dataset](https://drive.google.com/drive/folders/1hpA_zdLswchUfw2x6N9jMzboc-paiNw-?usp=drive_link).    
To train classification, we divided datasets into 5 classes as we mentioned.    
To make more dataset, we augmented datasets using changing HSV, brightness, contrast.. and so on.    

## Swimming pretrained models
To infer, you can use this [pretrained model](https://drive.google.com/drive/folders/1OUAETnTg0SkwO5NA7QIU3t3F4I9QIRYI?usp=drive_link).    
It contained original FairMOT's swimming pretrained model (resnet18_epoch150.pth) and classification FairMOT head pretrained model (FairMOT_classhead_v12.pth).    


