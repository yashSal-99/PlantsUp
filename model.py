import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import sklearn
import os
# import cv2
#import keras
import tensorflow as tf
import tensorflow_hub as hub



IMG_SIZE = 300

# Create a function for preprocessing images
def process_images(image_path , img_size=IMG_SIZE):
  """
  Takes an image file path and turns the image into tensors.
  """
  # read in a image filepath
  image = tf.io.read_file(image_path) # return a tensor dtype string.

  # turns the jpeg image into numerical Tensor with # color channels(Red,Green,Blue)
  image = tf.image.decode_jpeg(image,channels=3) # takes argument as dtype string.

  # convert the colour channels values from 0-255 to 0-1 values(normalization)
  image = tf.image.convert_image_dtype(image, tf.float32)# can convert into types like uint32, uint64, int8, int16 ,float64, float32.

  # resize the image to our desired value(300,300)
  image = tf.image.resize(image, size=[img_size,img_size])

  return image

# Create a simple function to return a tuple(image,label)

def get_image_label_tuple(image_path, label):
  """
  takes an image file path name and th assosicated label, processses the image and returns a tuple of(image,label).
  """
  image = process_images(image_path)
  return image ,label


# Define the batch size,32 is good.
BATCH_SIZE = 32

#Create a function to turn data into batches
def create_data_batches(X,Y=None , batch_size=BATCH_SIZE, valid_data=False ,test_data=False):
  """
  Creates batches of data out of image(X) and label(Y) pairs.
  shuffles the data if it's training data but doesn't shuffle if it's validation data.
  Also accepts test data as input(no labels).
  """
  # If the data is a test dataset, we probably don't have labels
  if test_data:
    print("Creating test data batches..")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) #only filepaths(no labels)
    data_batch = data.map(process_images).batch(BATCH_SIZE)
    return data_batch

  #If the data s a valid dataset, we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X),  # filepaths
                                               tf.constant(Y)))# labels
    data_batch = data.map(get_image_label_tuple).batch(BATCH_SIZE)
  else :
    print("Creating training data batches...")
    # turn filepaths and labels into Tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X),
                                              tf.constant(Y)))
    #shuffling pathnames and labels before mapping image processor function is faster than shuffing processed images
    data = data.shuffle(buffer_size=len(X))

    #Create (image,label) tuples(image to preprocessed image)
    data = data.map(get_image_label_tuple)

    # turn the training data into batches
    data_batch = data.batch(BATCH_SIZE)
  return data_batch

#Create a function to lad a model.
def load_a_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from:{model_path}...")
  model = tf.keras.models.load_model(model_path,custom_objects={"KerasLayer":hub.KerasLayer})
  return model

# Turning this into function.

unique_labels = ['Aloevera', 'Amla', 'AmrutaBalli', 'Arali', 'Ashoka', 'Ashwagandha',
 'Astma_weed', 'Avacado', 'Badipala', 'Balloon_Vine', 'Bamboo', 'Basale',
 'Beans', 'Betel', 'Betel_Nut', 'Brahmi', 'Bringaraja', 'Caricature', 'Castor',
 'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee',
 'Coriender', 'Curry', 'Doddapatre', 'Drumstick', 'Ekka', 'Eucalyptus',
 'Ganigale', 'Ganike', 'Gasagase', 'Geranium', 'Ginger', 'Globe Amarnath',
 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine',
 'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemon_grass',
 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Nagadali',
 'Neem', 'Nelavembu', 'Nerale', 'Nithyapushpa', 'Nooni', 'Onion', 'Padri',
 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomegranate',
 'Pumpkin', 'Raddish', 'Raktachandini', 'Rose', 'Sampige', 'Sapota',
 'Seethaashoka', 'Seethapala', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato',
 'Tulsi', 'Tulsi (1)', 'Turmeric', 'Wood_sorel', 'camphor', 'kamakasturi',
 'kepala']




def get_pred_label(predictions):
  """
  Finding maximum predicted label for a prediction.Turns an array of prediction probabilities into a label.
  """
  return unique_labels[np.argmax(predictions)]



model = load_a_model("25122024-120825-full-data-efficientnetb3_adam_model.h5")


# Testing input image.
#input_image = ['11.jpg']

def predict_plant_species(input_image):
  test_data = create_data_batches(input_image, test_data=True)

  predictions = model.predict(test_data,verbose =1)

  return(get_pred_label(predictions))










