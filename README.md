# CIE XYZ Net: Unprocessing Images for Low-Level Computer Vision Tasks
*[Mahmoud Afifi](https://sites.google.com/view/mafifi)*, *[Abdelrahman Abdelhamed](https://www.eecs.yorku.ca/~kamel/)*, *[Abdullah Abuolaim](https://sites.google.com/view/abdullah-abuolaim/)*, *[Abhijith Punnappurath](https://abhijithpunnappurath.github.io/)*, and  *[Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)*
<br></br>York University

Reference code for the paper [CIE XYZ Net: Unprocessing Images for Low-Level Computer Vision Tasks](https://arxiv.org/pdf/2006.12709.pdf). Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith Punnappurath, and Michael S. Brown, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021. If you use this code or our dataset, please cite our paper:
```
@article{CIEXYZNet,
  title={CIE XYZ Net: Unprocessing Images for Low-Level Computer Vision Tasks},
  author={Afifi, Mahmoud and Abdelhamed, Abdelrahman and Abuolaim, Abdullah and Punnappurath, Abhijith and Brown, Michael S},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  pages={},
  year={2021}
}
```


## Code (MIT License)
![network_design](https://user-images.githubusercontent.com/37669469/81250194-550b1700-8fee-11ea-8a69-0fde90f1062f.jpg)

<!-- PyTorch -->

<p align="left">
  <img width = 20% src="https://user-images.githubusercontent.com/37669469/81490764-0c549780-9254-11ea-813c-02de8da42102.png">
</p>

#### Prerequisite
1. Python 3.6
2. opencv-python
3. pytorch (tested with 1.5.0)
4. torchvision (tested with 0.6.0)
5. cudatoolkit
6. tensorboard (optional)
7. numpy 
8. future
9. tqdm
10. matplotlib

##### The code may work with library versions other than the specified.

#### Get Started

#### Demos:
1. Run `demo_single_image.py` or `demo_images.py` to convert from sRGB to XYZ and back. You can change the task to run only one of the inverse or forward networks.
2. Run `demo_single_image_with_operators.py` or `demo_images_with_operators.py` to apply an operator(s) to the intermediate layers/images. The operator code should be located in the `pp_code` directory. You should change the code in `pp_code/postprocessing.py` with your operator code. 

#### Training Code:
Run `train.py` to re-train our network. You will need to adjust the training/validation directories accordingly. 

#### Note:
All experiments in the paper were reported using the Matlab version of CIE XYZ Net. The PyTorch code/model is provided to facilitate using our framework with PyTorch, but there is no guarantee that the Torch version gives exactly the same reconstruction/rendering results reported in the paper. 

<br></br>

<!-- Matlab -->

<p align="left">
  <img width = 25% src="https://user-images.githubusercontent.com/37669469/81493516-e1c40800-926e-11ea-8685-11f41ade7ed4.png">
</p>

#### Prerequisite
1. Matlab 2019b or higher 
2. Deep Learning Toolbox

#### Get Started
Run `install_.m`. 

#### Demos:
1. Run `demo_single_image.m` or `demo_images.m` to convert from sRGB to XYZ and back. You can change the task to run only one of the inverse or forward networks.
2. Run `demo_single_image_with_operators.m` or `demo_images_with_operators.m` to apply an operator(s) to the intermediate layers/images. The operator code should be located in the `pp_code` directory. You should change the code in `pp_code/postprocessing.m` with your operator code. 

#### Training Code:
Run `training.m` to re-train our network. You will need to adjust the training/validation directories accordingly. 
<br></br>



## sRGB2XYZ Dataset
![srgb2xyz](https://user-images.githubusercontent.com/37669469/80854947-4eedf280-8c0a-11ea-8ada-e12bea63bdc6.jpg)

Our sRGB2XYZ dataset contains ~1,200 pairs of camera-rendered sRGB and the corresponding scene-referred CIE XYZ images (971 training, 50 validation, and 244 testing images).

Training set (11.1 GB): [Part 0](https://ln2.sync.com/dl/a2894dbb0/sp365wf7-rtd9tujt-kaqqpcpq-mnpph44z) | [Part 1](https://ln2.sync.com/dl/d55a95be0/zg95xg6u-n8nf7kc5-pttv6z8f-n4yu3yny) | [Part 2](https://ln2.sync.com/dl/fb406ca40/j5wmbqdx-knia8qia-cm9yisub-mjcmmbjy) | [Part 3](https://ln2.sync.com/dl/508d5e380/tyhx4efv-ibirjzzu-vid3hjdr-m4j2yxan) | [Part 4](https://ln2.sync.com/dl/e0941e650/hsu3z5dp-fa5ird2b-uiv8tqjy-nfq6cje6) | [Part 5](https://ln2.sync.com/dl/258b02190/9jmarz63-ct33xx4e-x4ikhfwt-guan99b7) 

Validation set (570 MB): [Part 0](https://ln2.sync.com/dl/de4bc8380/xiughx76-cf6xcbp4-vzr73pde-3iyf4spk) 

Testing set (2.83 GB): [Part 0](https://ln2.sync.com/dl/bb19d1b90/nv38zdmq-n4b46kgq-hv7sj472-dfxbzz2u) | [Part 1](https://ln2.sync.com/dl/17f046300/5qcidmk6-rqhqqy57-dwybi55v-f8kz9xku)

### Dataset License:
As the dataset was originally rendered using raw images taken from the [MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/), our sRGB2XYZ dataset follows the original license of the [MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/).

