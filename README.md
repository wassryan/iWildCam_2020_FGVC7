# iWildCam_2020 FGVC7
Top 3% (3/126)  solution for [iWildCam 2020](https://www.kaggle.com/c/iwildcam-2020-fgvc7?rvi=1) competition (Categorize animals in the wild), which is a part of the  *FGVC7* workshop at *CVPR 2020*

### Requirements
* Python 3.6
* pytorch 1.4.0

### About the Code

#### 1. Prepare Data
Download the competition data from [kaggle website](https://www.kaggle.com/c/iwildcam-2020-fgvc7/data)

*crop data*
```
python fast_crop_image.py # crop data from images
```
*prepare json for train/val*

```
python prepare_data.py
```
*prepare json for test data*
```
python sort_images.py
```

#### 2. Train the Model
1. for classification model(e.g. *resnet, resnext, efficientnet...*)
```
python train_model224.py -cfg configs/efficientNet.yaml
```
2. for NTS model
```
python train.py
```
#### 3. Prediction

```
python infer224.py/infer.py
```

#### 4. Train Cross Model
if you want to train K-cross validation model, and infer it
1. use gen_kcross.py to create kcross train.json/val.json
2. then train and infer it 
```
# first set CROSS_VALIDATION as True in xxx.yaml, then 
python train_model224.py
python infer_crossmodel.py
```
#### 5. Ensemble models
use `model_ensemble.py`

### What Have not been released

1. data sample used for long-tail methods
2. auxiliary classifier head(for location)


### Result Table

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">train epochs</th>
<th valign="bottom">Private Score.</th>
<th valign="bottom">Public Score.</th>
<!-- TABLE BODY -->
<tr><td align="left">EfficientNet-B0</td>
<td align="center">36</td>
<td align="center">83.6</td>
<td align="center">82.6</td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">36</td>
<td align="center">83.5</td>
<td align="center">82.3</td>
</tr>
<tr><td align="left">NTS-Net</td>
<td align="center">36</td>
<td align="center">84.6</td>
<td align="center">84.0</td>
</tr>
<tr><td align="left">SEResnext101</td>
<td align="center">36</td>
<td align="center">82.6</td>
<td align="center">82.8</td>
</tr>
<tr><td align="left">Ensemble</td>
<td align="center">36</td>
<td align="center">84.7</td>
<td align="center">84.5</td>
</tr>
</tbody></table>