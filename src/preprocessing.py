import os
import pandas as pd
from utils import *

dataset_path = "./../PlantVillage/"
data = []
labels = []
specie = []
disease = []
i=0
image_label_dict = {}

# Iterate through the dataset directory
for class_name in os.listdir(dataset_path):
  if(class_name!=".DS_Store"):
    class_dir = os.path.join(dataset_path, class_name)
    for img_name in os.listdir(class_dir):
        if img_name !=".DS_Store":
            img_path = os.path.join(class_dir, img_name)
            data.append(img_path)
            specie.append(class_dir.split("/")[-1].split("___")[0])
            disease.append(class_dir.split("/")[-1].split("___")[1])
            labels.append(i)
    image_label_dict[i] = class_dir.split("/")[-1]
    i+=1

df = pd.DataFrame({'data': data, 'label': labels,'specie':specie,'disease':disease})

# Create datasets
train_df, val_df,test_df = create_train_validation_split(df, 0.3, "data", 142)

for th in [-1,178,174,89,44]:
    val_images, val_labels = transform_dataset(val_df,th)
    train_images, train_labels = transform_dataset(train_df,th)
    test_images, test_labels = transform_dataset(test_df,th)
    np.save('../input/val_images_'+str(th)+'.npy', val_images)
    np.save('../input/val_labels_'+str(th)+'.npy', val_labels)
    np.save('../input/train_images_'+str(th)+'.npy', train_images)
    np.save('../input/train_labels_'+str(th)+'.npy', train_labels)
    np.save('../input/test_images_'+str(th)+'.npy', test_images)
    np.save('../input/test_labels_'+str(th)+'.npy', test_labels)