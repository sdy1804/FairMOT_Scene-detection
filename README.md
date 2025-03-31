# FairMOT-Scene-detection

## Introduction
In swimming competition video, there are some scenes that repeatly occured when swimmers do competition.     
I classified some scenes - On-block, Diving, Swimming, Turning, Finish.    
Optimized hyper-parameters were different for each classes, so I utilized these classes.    
After classification head classified scene classes, hyper-parameters are changed by each optimized values for classes.     
It can increase swimmer tracking persistance and MOTA score.    
Hyper-parameters can be Position, Velocity for Kalman filter and aspect ratio's weight for cIoU function.    
I changed IoU function that used in IoU matching stage to cIoU function.    
cIoU function can consider 

Swimming scene detection dataset

https://drive.google.com/drive/folders/1hpA_zdLswchUfw2x6N9jMzboc-paiNw-?usp=drive_link


Swimming pretrained models (FairMOT, Scene detection FairMOT head)

https://drive.google.com/drive/folders/1OUAETnTg0SkwO5NA7QIU3t3F4I9QIRYI?usp=drive_link
