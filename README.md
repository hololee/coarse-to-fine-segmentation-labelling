# CFS-Label: A Pixel-Level Coarse-to-Fine Segmentation Labelling Algorithm

This repo only explains the mechanism of [this paper](http://)   
Just core scripts are extracted by origin project.

~~~
Whole project cannot be attached this repo because of some project contract.  
So if you want to get more information, contact to e-mail.
~~~

## Environments
See `requirements.txt`

## Models
Two type of model included in `./models`

1. ``./model/UNet_.py``  
**UNet** model using transpose convolution.


2. ``./model/UNet_weight_connection.py``  
**UNet** model using transpose convolution with weighted skip connection.


## Architecture  
![whole architecture](./imgs/arch.png)


## Result   
### 1. bean leaf result   
![bean result](./imgs/slice3.png)   

### 2. paprika disease result   
![paprika result](./imgs/slice4.png)
