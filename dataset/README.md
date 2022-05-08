# Kinetics-400 Dataset 

This repository consists of a small size of Kinetics-400 dataset so that understanding the 'data preparation process, and working of PyTorch dataset and dataloader.

If you want to achieve full size, please refer to below:

* Download the dataset from [DoranLyong](https://github.com/DoranLyong)/**[kinetics-dataset](https://github.com/DoranLyong/kinetics-dataset)**. 
* The Kinetics project publications can be found here: https://deepmind.com/research/open-source/kinetics.



***

## Mini Kinetics-400 

### preliminaries

* reference to [YutaroOgawa's github](https://github.com/YutaroOgawa/pytorch_advanced/tree/master/9_video_classification_eco/video_download) ← for minimal Kinetics-400  
* reference to [ActivityNet/Crawler/Kinetics](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics) ← baseline with python2 execution



<b>[update]</b> - for python3 

The previous execution(with python2) is expired, so we need to follow like here; [Update environment.yml](https://github.com/activitynet/ActivityNet/pull/73). 

* go to [LukasHedegaard's github](https://github.com/LukasHedegaard/ActivityNet/tree/update-kinetics-crawler-environment/Crawler/Kinetics), then achieve both ```download.py```  and ```environment.yml``` . 

* create your environment by running 

  ```bash
  conda env create -f environment.yml
  source activate kinetics
  ```

* update python packages 

  ```bash
  pip install --upgrade joblib 
  pip install --upgrade youtube-dl
  ```

  

### Usage 

```bash
python run_download.py
```



If everything is successful, you should get 8 video clips in ```kinetics_video``` directory. 
