#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:30:17 2018

@author: xhuo
"""



import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.layers import Input,Conv2D,UpSampling2D,Dense,Reshape,BatchNormalization,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE



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


# evaluate model by visualizing training history and loss on test set
def evaluate_model(history,model,x_test):
    """
    Arguments:
        history (History): returned object by the fit method of models. it is a 
                           record of training loss values and metrics values at 
                           successive epochs, as well as validation loss values
                           and validation metrics values (if applicable).
        model(Model) : the model trained to evaluate
        x_test(array) : test set
    """
    # list all data in history
    print(history.history.keys())
    #Visualize history (loss vs epochs)
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model_loss')
    plt.ylabel('loss') 
    plt.xlabel('epochs')
    plt.legend(['train_loss','val_loss'], loc='upper left')
    plt.show()
    # test on test set to get test loss
    score = model.evaluate(x_test, x_test, verbose=0)
    print('Test loss:', score)

# visualize three kinds of images: original images, latent space and resulting images
def test_model(model,x_test):
    """
    Arguments:
        model(Model) : the model trained to evaluate
        x_test(array) : test set
    """
    # Extract features from the layer "encoder_output"
    get_encoded = Model(inputs= model.input, outputs= model.get_layer('encoder_output').output) 
    #latent space
    x_encoded = get_encoded.predict(x_test)
    encoder_nor =[]
    for i in x_encoded:
        x = Normalization(i)
        encoder_nor.append(x)
    encoder_nor = np.array(encoder_nor)
    # the output of autoencoder
    decoded_imgs = model.predict(x_test)
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

# normalize the data
def Normalization(x):
    """
    Arguments:
        x(array)
    Returns :
        array normalized
    """
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

# reduce dimension and visualize the data
def tsne(model,x_train):
    """
    Arguments:
        model(Model) : the model trained to evaluate
        x_train(array) : training set
    """
    get_encoded = Model(inputs= model.input, outputs= model.get_layer('encoder_output').output)
    x_encoded = get_encoded.predict(x_train)
    # reduce dimension to 2 
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(x_encoded)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], label="t-SNE")
    plt.legend()
    plt.show() 
    
    
#Creat Model
def build_model_VGG16AE(dim,lr,TRAINABLE=False,BN = False):
    """
    Arguments:
        dim(int) : the dimension of latent space
        lr(float) : the learning rate for training
        TRAINABLE(boolean): freeze layers or not
        BN(boolean) : add the layer of BatchNormalization or not
    Returns:
        autoencoder(model) : the created model
    """
    
    # input shape :224*224*3
    input_image = Input(shape = (224, 224, 3))
    # download vgg16 model trained on imagenet
    base_model = VGG16(weights='imagenet')
    
    # freeze all layers of base_model or not
    for layer in base_model.layers:
        layer.trainable=TRAINABLE
    #-------------------encoder---------------------------- 

    #    block1
    encoder = base_model.get_layer('block1_conv1')(input_image)
    if BN : encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block1_conv2')(encoder)
    if BN : encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block1_pool')(encoder)
    
    #    block2
    encoder = base_model.get_layer('block2_conv1')(encoder)
    if BN : encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block2_conv2')(encoder)
    if BN : encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block2_pool')(encoder)
    
    #    block3
    encoder = base_model.get_layer('block3_conv1')(encoder)
    if BN : encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block3_conv2')(encoder)
    if BN : encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block3_conv3')(encoder) 
    if BN : encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block3_pool')(encoder)
    
    #    block4
    encoder = base_model.get_layer('block4_conv1')(encoder)
    if BN : encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block4_conv2')(encoder)
    if BN : encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block4_conv3')(encoder)
    if BN : encoder = BatchNormalization()(encoder) 
    encoder = base_model.get_layer('block4_pool')(encoder)
    
    #    block5
    encoder = base_model.get_layer('block5_conv1')(encoder)
    if BN : encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block5_conv2')(encoder)
    if BN : encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block5_conv3')(encoder)
    if BN : encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block5_pool')(encoder)     
    
    #  block 6
    encoder = Flatten()(encoder)
    encoder = base_model.get_layer('fc1')(encoder)
    if BN : encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('fc2')(encoder)
    if BN : encoder = BatchNormalization()(encoder)
   
    #--------latent space (trainable) ------------
    encoder = Dense(dim, activation='relu', name = 'encoder_output')(encoder)
    #encoder = Dense(32,name = 'encoder_output')(encoder)     
    #--------------decoder (trainable)-----------   
    # Block 6
    decoder = Dense(4096, activation='relu', name = 'dblock_dense1')(encoder)
    if BN : decoder = BatchNormalization()(decoder)
    decoder = Dense(4096, activation='relu', name = 'dblock_dense2')(decoder)
    if BN : decoder = BatchNormalization()(decoder)
    decoder = Dense(7*7*512, activation='relu', name = 'dblock_dense3')(decoder)
    if BN : decoder = BatchNormalization()(decoder)
    decoder = Reshape([7,7,512],name = 'dblock_reshpe')(decoder)    
    decoder = UpSampling2D((2,2))(decoder)        
    
    # Block 5
    decoder = Conv2D(512, (3, 3), activation='relu', padding='same', name='dblock5_conv1')(decoder)
    if BN : decoder = BatchNormalization()(decoder)
    decoder = UpSampling2D((2,2))(decoder)
    
    # Block 4
    decoder = Conv2D(256, (3, 3), activation='relu', padding='same', name='dblock4_conv1')(decoder)
    if BN : decoder = BatchNormalization()(decoder)
    decoder = UpSampling2D((2,2))(decoder)     
         
    # Block 3
    decoder = Conv2D(128, (3, 3), activation='relu', padding='same', name='dblock3_conv1')(decoder)
    if BN : decoder = BatchNormalization()(decoder)
    decoder = UpSampling2D((2,2))(decoder)        
     
    # Block 2
    decoder = Conv2D(64, (3, 3), activation='relu', padding='same', name='dblock2_conv1')(decoder)
    if BN : decoder = BatchNormalization()(decoder)
    decoder = UpSampling2D((2,2))(decoder) 
    
    # Block 1
    decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='dblock1_conv1')(decoder)
  
       
    autoencoder = Model(input_image, decoder)
    #autoencoder.compile(loss='mse', optimizer = Adam())
    autoencoder.compile(loss='binary_crossentropy', optimizer = Adam(lr=lr, beta_1=0.95, beta_2=0.999))
    # visualize model
    autoencoder.summary()
    
    return autoencoder


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained"
    parser.add_argument("-m", "--model", help=help_)
    help_ = "the dimension of latent space"
    parser.add_argument("-d", "--dimension", type=int, default=32, help=help_)
    help_ = "learning rate"
    parser.add_argument("-r", "--rate", type=float, default=0.0001, help=help_)
    help_ = "epoch for training"
    parser.add_argument("-e", "--epoch", type=int, default=50, help=help_)
    help_ = "the path for saving model"
    parser.add_argument("-p", "--path", help=help_)
    args = parser.parse_args()
    
    
    
    # Load images : training set and test set
    
    x_train = load_image('dbcafe/x_train/')
    x_test = load_image('dbcafe/x_test/')
    
    print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_train[0,:,:,:].shape, 'image size')
    
    x_train = x_train.astype('float32')
    x_train = x_train/255.
    x_test = x_test.astype('float32')
    x_test = x_test/255.
    
    
    if args.model:
        # load model trained
        print('load model')
        ae = load_model(args.model)
        print('finish loading model')
        
        #test model
        test_model(ae,x_test)
        #tsne
        tsne(ae,x_train)
        
    
    if args.path: 
        #train model
        ae = build_model_VGG16AE(args.dimension,args.rate,BN = True)
        # split training set and validation set
        x_train, x_valid = train_test_split(x_train, test_size=0.2, random_state=123)
        history = ae.fit(x_train, x_train, batch_size=32,
                          epochs=args.epoch,verbose=1,validation_data = (x_valid,x_valid))
        print('finish training')
        # save model
        ae.save(args.path)
        #evaluate model
        evaluate_model(history, ae, x_test)
        #test model
        test_model(ae,x_test)
        #tsne
        tsne(ae,x_train)

    





