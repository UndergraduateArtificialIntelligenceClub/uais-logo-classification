import pandas as pd
import numpy as np
import os 
import glob

def rename_logo_images(folder_name = "Images/uais_logo_images"): 
    uais_logo_images = glob.glob(os.path.join(folder_name, '*.jpeg'))
    for i, image_path in enumerate(uais_logo_images):
        new_image_name = os.path.join(folder_name, f'{i}.jpeg')

        os.rename(image_path, new_image_name)

if __name__ == "__main__": 

    rename_logo_images()