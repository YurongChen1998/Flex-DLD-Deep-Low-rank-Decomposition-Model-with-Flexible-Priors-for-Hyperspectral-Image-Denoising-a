# Flex-DLD: Deep Low-rank Decomposition Model with Flexible Priors for Hyperspectral Image Denoising and Restoration

- ***Denoising Data Preparation***

1) Datasets for hyperspectral images **denoising** experiments include: [KAIST](https://drive.google.com/drive/folders/1EmTZoOkfKnPQHfMCxcYyGH9uJJxzAz_5?usp=sharing), [Washington DC Mall](https://drive.google.com/drive/folders/12QBJk2EvaqjEdd5hNGPICzunT3NZk2XZ?usp=sharing), [CAVE](https://drive.google.com/drive/folders/1MwhGEpO6BzZYZkKtIME-mkFwgdIBBNbL?usp=sharing), [Indian Pines](https://drive.google.com/drive/folders/1vAkEki8JQMP3cavSVIf27b5XVOVudFjm?usp=drive_link)


You can download in: [Google drive](https://drive.google.com/drive/folders/1y9wa5fv87D73zW-F2N-5_wK3qPVLq12E?usp=sharing) or [Baidu drive](https://pan.baidu.com/s/1NC-NcqVTR1yIFZyaE0aNOg) (Code: emg2)

```
Put the downloaded data into the [Data] folder

Alternatively, you can generate noisy hyperspectral images according to the code
```

2) Denosing your hyperspectral images

```
Put your data into the [Data] folder
The mat file should include [noisy_img] and [img] variables
```

- ***Denoising Experiments**

```
cd Denoise
python main_*.py
(We have uploaded KAIST Scene01 data. You can run main_1_KAIST.py directly.)
```

- ***Check Our Denoising Results***
  
The denoised hyperspectral images of KAIST, WDC, CAVE, and IP datasets are available here: [Baidu drive](https://pan.baidu.com/s/1vGzzZmltadKDqTf2CPkWxw) (Code: 4g02)







---
- ***Restoration Data Preparation***
  
Datasets for hyperspectral images **restoration** experiments include: [KAIST](https://drive.google.com/drive/folders/1f_vAYwCmXp1kNcg54yLO145I_tzdpwIs?usp=drive_link)


You can download in: [Google drive](https://drive.google.com/drive/folders/1f_vAYwCmXp1kNcg54yLO145I_tzdpwIs?usp=drive_link) or [Baidu drive](https://pan.baidu.com/s/1mdLWXgvzkmQscfZu4t4M7A) (Code: u4er)

```
Put the downloaded data into the [Data] folder
```

- ***Restoration Experiments***

```
cd Denoise/CASSI_Restoration/
python main.py
(We have uploaded KAIST Scene01 data. You can run main.py directly.)
```

- ***Check Our Reconstructed Results***
  
The reconstructed hyperspectral images of KAIST datasets are available here: [Baidu drive](https://pan.baidu.com/s/1LJ0W7QiTajowGZp_XCVVRw ) (Code: 87hh)







---
- ***Code Description***
```
main_*.py               : code for denoising
optimization.py         : code of ADMM iteration
func.py                 : code includes some useful functions
test_metric             : code for evaluation
model/model_loader.py   : code for loading deep low-rank networks
model/LRNet.py          : code for designing network architecture
model/common.py         : code includes some network blocks
model/basicblock.py     : code includes some network blocks
model/utils.py          : code includes functions for evaluation
```
