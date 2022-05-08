import os
from pathlib import Path 



data_dir = f'kinetics_videos'
Path(data_dir).mkdir(parents=True, exist_ok=True)


os.system(f"python ./download.py ./anno/kinetics-400_val_8videos.csv ./{data_dir}")
