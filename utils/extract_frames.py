""" code reference to 
    - https://github.com/DoranLyong/nvGesture-tutorial/blob/main/utils/nv_extract_frames.py
"""
import os 
import os.path as osp 
import glob 
from pathlib import Path 
from subprocess import call

import click


@click.command()
@click.option('--project_root', required=True, 
                default=osp.join('/home/milky/Workspace/Action_Recognition_Study/tutorials/Kinetics-400-tutorial'),
                help="Root path for dataset")
@click.option('--dataset_path', required=True,
                default=osp.join("dataset", "kinetics_videos"), 
                help="dataset path"
            )
def main(project_root:str,
         dataset_path:str
        
        ):
    click.echo(f"project_root: {project_root}")
    click.echo(f"dataset_path: {dataset_path}")

    dataset_root = osp.join(project_root, dataset_path)
    class_list = os.listdir(dataset_root)
    print(f"class_list: {class_list}")
    

    # Extract frames 
    # -----------------
    extract_frames(dataset_root, class_list)    


# =================
# -----------------
def extract_frames(dataset_root:str, class_list:list):
    """ Extract frames of .avi files. 
    """
    for cls_item in class_list:
        
        cls_path = osp.join(dataset_root, cls_item)
        files = glob.glob(osp.join(cls_path, '*.mp4')) # process only .mp4 format 

        for file in files: 
            print(f"Extracting frames for {file}")

            # split into 'file_name' and 'extension' 
            name, ext = osp.splitext(file)

            # path for saving frames 
            Path(name).mkdir(parents=True, exist_ok=True)

            # extract frames 
            call(["ffmpeg", "-i",  file, "-vf", "scale=-1:256" ,osp.join(name, "%05d.png"), "-hide_banner"]) 

            



if __name__ == "__main__":
    main() 