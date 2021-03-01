import pandas as pd
import os, shutil

# uses pandas to load train csv
train_df = pd.read_csv('/mnt/vol_b/datasets/cassava-leaf-disease-classification/train.csv')
train_df.set_index('image_id', inplace=True)

# copies images to relevant labeled folders
try:
    os.mkdir(f'/mnt/vol_b/datasets/cassava-leaf-disease-classification/train_images_sorted')
    for i in range(5):
        os.mkdir(f'/mnt/vol_b/datasets/cassava-leaf-disease-classification/train_images_sorted/{i}')
except OSError as error:
    print(error)
    pass

train_img_dir = '/mnt/vol_b/datasets/cassava-leaf-disease-classification/train_images'

for imgname in os.listdir(train_img_dir):
    imglabel = train_df.loc[imgname][0]
    shutil.copy(f'{train_img_dir}/{imgname}', f'/mnt/vol_b/datasets/cassava-leaf-disease-classification/train_images_sorted/{imglabel}')
