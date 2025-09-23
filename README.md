# FAO-Former
FAO-Former clarifies the modeling mechanism of query representations and the decision-making process of the model.
<img width="1135" height="540" alt="image" src="https://github.com/user-attachments/assets/440db97a-2763-423d-a658-38e2b5aa85e3" />

# Reqirements
```

pytorch = 2.3.1
python = 3.9.19
cuda = 11.3
numpy = 1.26.4
mmcv = 2.1.0
mmengine = 0.10.3
mmsegmentation = 1.2.2
grad-cam = 1.5.4

```

# How to use?

1. Download the [Cityscapes](https://www.cityscapes-dataset.com/).
2. Install Reqirements
3. Edit the data path
4. Train and Evaluation

# Dataset
Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset.

# Train

To train FAO-Former, run the training script below.

```
python ./train.py ${CONFIG_FILE} --resume --cfg-options load_from=${CHECKPOINT}
```

For instance：

```
python ./train.py configs/FAO_former_r101_MSDFDConvPD_8xb2-60epoch_cityscapes-512x1024.py
```

# Test

To test FAO-Former, run the testing script below.
    
```
python ./test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```  

For instance(Download checkpoint: https://pan.baidu.com/s/1l93sSqKXTSlQbPUE88voDw   passwd：ihek)：

```
python ./test.py configs/FAO_former_r101_MSDFDConvPD_8xb2-60epoch_cityscapes-512x1024.py checkpoints/ckpt_FAO_former_r101_Cityscapes.pth
```

## Acknowledgement

We utilized code from:

- [openmmlab segmentation](https://mmsegmentation.readthedocs.io/en/latest/) 
- [pytorch_grad_cam](https://github.com/jacobgil/pytorch-grad-cam/tree/61e9babae8600351b02b6e90864e4807f44f2d4a)  

Thanks for their wonderful works.



