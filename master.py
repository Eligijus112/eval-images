"""
The main script that controls the pipeline
"""

### Loading modules

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import cv2
import os
import re

### Loading custom functions

def read_NN_model(path_specs, path_weights,
                  loss_f = 'sparse_categorical_crossentropy', 
                  acc_metric = 'accuracy'):
    """
    Reads the specification and the weights from saved files. 
    Additionaly, compiles the model for instant use.
    """
    json_file = open(path_specs, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_weights)
    loaded_model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss = loss_f,
              metrics = [acc_metric])
    return loaded_model 
    
def extract_max_row(image_fit_df, max_column):
    """
    A very custom function that extracts the image label with the 
    highest probability. The probability is an outout of the neural network 
    model fit.
    """
    all_images = set(image_fit_df['image_nr'])
    all_images = list(all_images)
    
    results = []
    for img in all_images:
        subset = image_fit_df[image_fit_df['image_nr'] == img]
        subset = subset[subset[max_column] == max(subset[max_column])]
        results.append(subset.index.values)
    return image_fit_df[image_fit_df.index.isin(results)]       

def construct_fit_frame(fit, decoder_frame):
    """
    Construcs a frame that augments the information from the fited keras model
    """
    index = range(1, len(fit)+1)
    fit_df = pd.DataFrame(fit, index=index)
    fit_df['image_nr'] = index
    fit_df = fit_df.melt(id_vars = 'image_nr', var_name = 'class_code', 
                         value_name = 'p')
    fit_df['class_code'] = fit_df.class_code.astype(int)
    decoder_frame['class_code'] = decoder_frame.class_code.astype(int)
    fit_df = fit_df.merge(decoder_frame, on = 'class_code')
    return fit_df


def create_path_frame(path, append_path = True, return_mapper = False):
    """
    Creates a dataframe with the links to the images in the *path* folder.  
    
    path (str): path to the folder where the images are
    append_path (bool): should we add the full path to the image?
    return_mapper (bool): should we return the image number?
    """
    all_photo = os.listdir(path)
    good_img = []
    for f in all_photo:
        if append_path: 
            f = path + '/' + f
        if re.search(r'.jpg$|.png$|.jpeg$', f) is not None:
            good_img.append(f)
    
    result = pd.DataFrame(good_img)
    result = result.rename({0 : 'path'}, axis = 'columns')
    
    if(return_mapper):        
        result['image_nr'] = range(1, len(good_img)+1)      
        result = result.rename({0 : 'image_nr'}, axis = 'columns')
    
    return result
    

def img_read(path, h, w, to_grey = True):
    """
    Reads and preproces an image in *path* 
    
    h (float): height of the resized image
    w (float): width of the resized image.
    to_grey (bool): should the image be grey scale?
    """
    img = cv2.imread(path)
    img = cv2.resize(img, (h, w))
    if(to_grey):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255    
    return img   

### Reading the class decoder

class_df = pd.read_csv('main_model/class_decoder.csv')

### Loading the model that is used in production 

main_model = read_NN_model('main_model/model_specs.json', 
                           'main_model/model_weights.h5')

### Reading and preprocesing all the photos

all_photo = create_path_frame('input', return_mapper = True) 
if all_photo.empty is not True:
    d = [img_read(x, h = 28, w = 28) for x in all_photo['path']]
    d = np.asarray(d)
    
    ### Predicting the image label probabilities
    
    fit = main_model.predict(d)
    
    ### Constructing a data frame to store the results in 
    
    fit_df = construct_fit_frame(fit, class_df)
    fit_df = fit_df.merge(all_photo, on = 'image_nr')
    fit_df = fit_df.sort_values(['image_nr'], ascending = True)
    
    ### Saving the results
    
    fit_df.to_csv('output/fitted_clases.csv', index = False)

