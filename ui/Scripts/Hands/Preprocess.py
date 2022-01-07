import os
import cv2
import random
import numpy as np
from tqdm import tqdm
num_images=int(input("Enter the number of images required for pre-processing"))
print(f"Preprocessing {num_images} from each category...")


# get all the directory paths
dir_paths = os.listdir("input/alphabet_valid")
dir_paths.sort()
root_path = 'input/alphabet_valid'


# get --num-images images from each category
for idx, dir_path  in tqdm(enumerate(dir_paths), total=len(dir_paths)):
    all_images = os.listdir(f"{root_path}/{dir_path}")
    os.makedirs(f"input/preprocessed_image_V/{dir_path}", exist_ok=True)
    for i in range(num_images): # how many images to preprocess for each category
        # generate a random id between 0 and 2999
        rand_id = (random.randint(0, 1499))
        image = cv2.imread(f"{root_path}/{dir_path}/{all_images[rand_id]}")
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(f"input/preprocessed_image_V/{dir_path}/{dir_path}{i}.jpg", image)
print('DONE')