#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:51:11 2018

@author: xhuo
"""

import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.layers import Input
from tensorflow.keras.models import Model,load_model


# convert images to a numpy array
def load_image(path):
    """
    Arguments:
        path (string): the path of dataset
    Returns:
        data (array): pixel values of dataset
    """
    files = glob.glob(path + '*png')
    data = []
    for f in files:
        img = cv2.imread(f)
        data.append(img)
    data = np.array(data)
    return data

# get the part of encoder, and the value to return is the latent space
def encoder(autoencoder,x):
    """
    Arguments:
        autoencoder (Model): autoencoder
    Returns:
        x (array): dataset
    """
    # Extract features from the layer "encoder_output"
    get_encoded = Model(inputs= autoencoder.input, outputs= autoencoder.get_layer('encoder_output').output)
    # latent space
    x_encoded = get_encoded.predict(x)
    return x_encoded

# visualize three kinds of images: original images, latent space and resulting images
def test_model(autoencoder,x_test):
    """
    Arguments:
        autoencoder(Model) : the model trained to evaluate
        x_test(array) : test set
    """
    #latent space
    x_encoded = encoder(autoencoder,x_test)
    encoder_nor =[]
    for i in x_encoded:
        x = Normalization(i)
        encoder_nor.append(x)
    encoder_nor = np.array(encoder_nor)
    # the output of autoencoder
    decoded_imgs = autoencoder.predict(x_test)
    n=10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        #display original
        ax = plt.subplot(3, n, i+1)
        plt.imshow(x_test[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        #display bottelneck
        ax = plt.subplot(3, n, i+n+1)
        plt.imshow(encoder_nor[i].reshape(8,4))
        plt.colorbar()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        #display reconstruction
        ax = plt.subplot(3, n, i+n*2+1)
        plt.imshow(decoded_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# find four nodes which the mean or variance are bigger    
def find(autoencoder,x_test):
    """
    Arguments:
        autoencoder(Model) : the model trained to evaluate
        x_test(array) : test set
    Returns :
        maxi(array) : the index of nodes whose mean or std is bigger
        x_encoded(vector): the latent vector
    """
    x_encoded = encoder(autoencoder,x_test)
    x = np.mean(x_encoded, axis = 0)
    n = 4
    maxi = []
    for i in range(n) :
      index = np.argmax(x)
      maxi.append(index)
      x[index] = 0.
    maxi = np.array(maxi)
    print(maxi)
    return maxi,x_encoded
    
    
# change four nodes to generate new images
def generator(maxi,decoder,x_encoded):
    """
    Arguments:
        decoder(Model) : the part of decoder
        maxi(array) : containing four index of nodes whose mean or std is bigger
        x_encoded(vector): the latent vector
    """
    n = 5
    img_size = 224
    sampling = x_encoded[1:2]
    figure = np.zeros((img_size * n, img_size * n, 3))
    
    for i in range(n):
        for j in range(n):
            # randomly sampled four nodes
            sampling[0,maxi[0]] = np.random.uniform(0,4)
            sampling[0,maxi[1]] = np.random.uniform(0,4)
            sampling[0,maxi[2]] = np.random.uniform(0,4)
            sampling[0,maxi[3]] = np.random.uniform(0,4)
            z_sample = sampling 
            x_decoded = decoder.predict(z_sample)
            img = x_decoded[0].reshape(img_size, img_size, 3)
            figure[i * img_size: (i + 1) * img_size,j * img_size: (j + 1) * img_size] = img

    plt.figure(figsize=(20, 20))
    plt.imshow(figure)
    plt.show()

# to plot images for linear transformation
def show(decoder,encoder_output):
    """
    Arguments:
        decoder(Model) : the part of decoder
        encoder_output(vector) : the latent vector
    """
    decoded_imgs = decoder.predict(encoder_output)
    # normalize the latent vector
    encoder_nor = []
    for i in encoder_output:
        x = Normalization(i)
        encoder_nor.append(x)
    encoder_nor = np.array(encoder_nor)
    print(encoder_nor.shape)
    
    n=11
    plt.figure(figsize=(20, 4))
    for i in range(n):
        #display bottelneck
        ax = plt.subplot(2, n, i+1)
        #plt.imshow(((encoder[i]* 255).astype(np.uint8)).reshape(8,4))
        plt.imshow(encoder_nor[i].reshape(8,4),cmap='Blues')
        plt.colorbar()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        #display reconstruction
        ax = plt.subplot(2, n, i+n+1)
        plt.imshow(decoded_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# normalize the data
def Normalization(x):
    """
    Arguments:
        x(array)
    Returns :
        array normalized
    """
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

# show the difference between two similar images
def differ(latent_0, latent_1,decoder):
    """
    Arguments:
        latent_0(array): the value of first image
        latent_1(array): the value of seconde image
        decoder(Model) : the part of decoder
    """
    diff = np.absolute(latent_1-latent_0)
    latent = []
    latent.append(latent_0)
    latent.append(diff)
    latent.append(latent_1)
    latent = np.array(latent)
    latent_nor = []
    for i in latent:
        x = Normalization(i)
        latent_nor.append(x)
    latent_nor = np.array(latent_nor)
    decoded_imgs = decoder.predict(latent)
    n=3
    plt.figure(figsize=(20, 4))
    for i in range(n):
        #display bottelneck
        ax = plt.subplot(2, n, i+1)
        #plt.imshow(((encoder[i]* 255).astype(np.uint8)).reshape(8,4))
        plt.imshow(latent_nor[i].reshape(8,4),cmap='Blues')
        plt.colorbar()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        #display reconstruction
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(decoded_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# create the part of decoder   
def decoder(base_model):
    """
    Arguments:
        base_model(Model): autoencoder
    Returns :
        decoder_model(Model): the part of decoder
    """
    
    latent_inputs = Input(shape=(32,), name='decoder_input') 
    # block 6
    decoder = base_model.get_layer('dblock_dense1')(latent_inputs)
    decoder = base_model.get_layer('dbc1')(decoder)
    decoder = base_model.get_layer('dblock_dense2')(decoder)
    decoder = base_model.get_layer('dbc2')(decoder)
    decoder = base_model.get_layer('dblock_dense3')(decoder)
    decoder = base_model.get_layer('dbc3')(decoder)
    decoder = base_model.get_layer('dblock_reshpe')(decoder)    
    decoder = base_model.get_layer('dblock_ups1')(decoder)        
    
    # Block 5
    decoder = base_model.get_layer('dblock_conv1')(decoder)
    decoder = base_model.get_layer('dbc4')(decoder)
    decoder = base_model.get_layer('dblock_ups2')(decoder)
    
    # Block 4
    decoder = base_model.get_layer('dblock_conv2')(decoder)
    decoder = base_model.get_layer('dbc5')(decoder)
    decoder = base_model.get_layer('dblock_ups3')(decoder)     
         
    # Block 3
    decoder = base_model.get_layer('dblock_conv3')(decoder)
    decoder = base_model.get_layer('dbc6')(decoder)
    decoder = base_model.get_layer('dblock_ups4')(decoder)        
     
    # Block 2
    decoder = base_model.get_layer('dblock_conv4')(decoder)
    decoder = base_model.get_layer('dbc7')(decoder)
    decoder = base_model.get_layer('dblock_ups5')(decoder) 
    
    # Block 1
    decoder = base_model.get_layer('dblock_conv5')(decoder)
    decoder_model = Model(latent_inputs, decoder,name = 'decoder')
    #visualize decoder model 
    decoder_model.summary()
    
    return decoder_model

if __name__ == '__main__':
    

    import argparse
    parser = argparse.ArgumentParser()
    help_ = "choose the method to analyze the latent space, 0 : change four nodes, 1: linear transformation"
    parser.add_argument("method",type=int, choices=[0, 1], help=help_)
    help_ = "the path for loading images"
    parser.add_argument("-p", "--path", help=help_, default = '.dbcafe/cardboard/')
    args = parser.parse_args()
    
    # load model trained
    base_model = load_model('./save_model/ae_sparse_cafe.h5')
    print('finish loading model')
    #create the part of decoder
    decoder = decoder(base_model)

    if args.model == 0 :
        #change four node to generate new images
        x_test = load_image(args.path) 
        x_test = x_test.astype('float32')
        x_test = x_test/255.
        # visualize the result
        test_model(base_model,x_test)
        # find four nodes whose mean or std is bigger
        maxi,x_encoded = find(base_model,x_test)
        #change four nodes to generate new images
        generator(maxi,decoder,x_encoded)
    
    
    if args.model == 1 :
         # compare two node   
        x_two = load_image('.dbcafe/two/') 
        x_two = x_two.astype('float32')
        x_two= x_two/255.
        # get the latent space of two images
        latent = encoder(base_model,x_two)
        hidden = []
        #first image
        latent_0 = np.array(latent[0])
        # seconde image
        latent_1 = np.array(latent[1])
        #tlinear transformation
        for i in range(11):
            x =  latent_0 + (latent_1-latent_0)*0.1*i
            hidden.append(x)
        hidden = np.array(hidden)
        # show the difference between two similar images
        differ(latent_0, latent_1,decoder)
        # plot images for linear transformation
        show(decoder,hidden)
        
      
     
    
  
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    