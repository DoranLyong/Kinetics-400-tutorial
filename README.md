# Kinetics-400 tutorial

* run by colab; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DoranLyong/Kinetics-400-tutorial/blob/main/notebook/Kinetics_400.ipynb)



## 1. Prerequisite 

* installing FFmpeg

```bash
sudo apt update 
sudo apt install ffmpeg 

# check 
ffmpeg -version 
```





## 2. Usage

<b>(step1)</b> Download the mini size dataset 

Go to ```dataset``` directory, and follow the instructions. 



<b>(step2)</b> Extract frames of video data.

```bash
python ./utils/extract_frames.py
```


## 3. Pytorch dataset & dataloader 
Check ```Video_DataLoader.ipynb```. 
