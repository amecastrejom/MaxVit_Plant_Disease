from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import pandas as pd
import numpy as np

def create_train_validation_split(df, test_size, target_column, random_state):
  """
  Creates train and validation splits from a DataFrame containing image paths.

  Args:
      df (pd.DataFrame): DataFrame with image paths in a column.
      test_size: Proportion of data for validation.
      target_column: Name of the column containing image paths.
      random_state: Seed for random splitting.

  Returns:
      tuple: A tuple of three DataFrames:
          - train_df: DataFrame containing training image paths.
          - val_df: DataFrame containing validation image paths.
          - test_df: DataFrame containing all image paths (unchanged).
  """

  # Get image paths
  all_paths = df[target_column].tolist()

  # Split data into training and validation sets
  train_paths, test_paths, train_label, test_label = train_test_split(df[target_column].tolist(),df["label"].tolist(), test_size=0.2, random_state=random_state,stratify=df["label"].tolist())
  train_paths, val_paths, train_label, val_label = train_test_split(train_paths,train_label, test_size=0.2, random_state=random_state,stratify=train_label)

  # Create DataFrames
  train_df = pd.DataFrame({'data': train_paths, 'label': train_label})
  val_df = pd.DataFrame({'data': val_paths, 'label': val_label})
  test_df = pd.DataFrame({'data': test_paths, 'label': test_label})

  return train_df, val_df, test_df

def transform_dataset(df,th):
  """
Modify the image size and resolution 

  Args:
      df (pd.DataFrame): DataFrame with image
      th (int): resolution of the images

  Returns:
      tuple: A tuple of two lists:
          - ds: List containing images as arrays
          - ds_label: List containing labels
  """
  ds = []
  ds_label = []
  for d,l in zip(df["data"],df["label"]):
      image = Image.open(d)
      if th>0:
        image.thumbnail((th,th))
      resized_image = cv2.resize(np.array(image), (224, 224))
      ds.append(resized_image)
      ds_label.append(l)
  return ds, ds_label