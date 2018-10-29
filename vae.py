#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:40:39 2018

@author: xhuo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:29:43 2018

@author: xhuo
"""

import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.layers import Input,Conv2D,UpSampling2D,Dense,Reshape,BatchNormalization,Flatten,Layer
from tensorflow.python.keras.layers import Lambda
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from keras import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop
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
    plt.title('modelloss')
    plt.ylabel('loss') 
    plt.xlabel('epochs')
    plt.legend(['train_loss','val_loss'], loc='upper left')
    plt.show()
    # test on test set to get test loss
    score = model.evaluate(x_test, None, verbose=0)
    print('Test loss:', score)

# visualize three kinds of images: original images, latent space and resulting images
def test_model(encoder_model,decoder_model,x_test):
    """
    Arguments:
        encoder_model(Model) : the part of encoder
        decoder_model(Model) : the part of decoder
        x_test(array) : test set
    """
    # the output of encoder
    mean,log_var,sampling = encoder_model.predict(x_test)
    # the output of decoder
    encoder_nor =[]
    for i in mean:
        x = Normalization(i)
        encoder_nor.append(x)
    encoder_nor = np.array(encoder_nor)
    decoded_imgs = decoder_model.predict(sampling)
    n=10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        #display original
        ax = plt.subplot(3, n, i+1)
        plt.imshow(x_test[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        #display bottelneck mean
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

# sample the latent space by using random to generate new images
def generator_image(decoder_model,dim):
    """
    Arguments:
        decoder_model(Model) : the part of decoder
        dim(int) : the dimesion of latent space
    """
    # display images generated from randomly sampled latent vector
    n = 10
    img_size = 224
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            z_sample = np.array([np.random.uniform(-1,1 ,size=dim)])
            x_decoded = decoder_model.predict(z_sample)
            img = x_decoded[0].reshape(img_size, img_size, 3)
            figure[i * img_size: (i + 1) * img_size,j * img_size: (j + 1) * img_size] = img

    plt.figure(figsize=(20, 20))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

#Reparameterization trick by sampling fr an isotropic unit Gaussian.
def sampling(args):
    """
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    from tensorflow.python.keras import backend as K
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# tsne : reduce dimension and visualize the data
def tsne(encoder_model,x_train):
    """
    Arguments:
        encoder_model(Model) : the model trained to evaluate
        x_train(array) : training set
    """
    x_encoded,_,_ = encoder_model.predict(x_train)
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(x_encoded)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], label="t-SNE")
    plt.legend()
    plt.show() 
    
# creat model
def build_model_VGG16VAE(dim,lr,TRAINABLE=False):
    """
    Arguments:
        dim(int) : the dimension of latent space
        lr(float) : the learning rate for training
        TRAINABLE(boolean): freeze layers or not
    Returns :
        encoder_model(Model) : the part of encoder
        decoder_model(Model) : the part of decoder
        vae(Model)
    """
    # input shape :224*224*3
    input_image = Input(shape = (224, 224, 3))
    # download vgg16 model trained on imagenet
    base_model = VGG16(weights='imagenet')
    # freeze all layers of base_model or not
    for layer in base_model.layers:
        layer.trainable=TRAINABLE
    #-------------------encoder---------------------------- 
    #--------(pretrained & trainable if selected)----------
    
     #    block1
    encoder = base_model.get_layer('block1_conv1')(input_image)
    encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block1_conv2')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block1_pool')(encoder)
        
    #    block2
    encoder = base_model.get_layer('block2_conv1')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block2_conv2')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block2_pool')(encoder)
    
    #    block3
    encoder = base_model.get_layer('block3_conv1')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block3_conv2')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block3_conv3')(encoder) 
    encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block3_pool')(encoder)
    
    #    block4
    encoder = base_model.get_layer('block4_conv1')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block4_conv2')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block4_conv3')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block4_pool')(encoder)
        
    #    block5
    encoder = base_model.get_layer('block5_conv1')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block5_conv2')(encoder)
    
    encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block5_conv3')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = base_model.get_layer('block5_pool')(encoder)     
    
    encoder = Flatten()(encoder)
    #encoder = Dense(1000,activation='relu', name = 'block_dense1')(encoder)
    #  mean vector
    z_mean = Dense(dim, name='z_mean')(encoder)
    z_mean = BatchNormalization()(z_mean)
    # standard deviation vector
    z_log_var = Dense(dim, name='z_log_var')(encoder)
    z_log_var = BatchNormalization()(z_log_var)
    #--------latent space (trainable) ------------
    latent = Lambda(sampling, output_shape=(dim,), name='latent')([z_mean, z_log_var])
        
    # instantiate encoder model
    encoder_model = Model(input_image, [z_mean, z_log_var, latent], name='encoder')
    encoder_model.summary()
    #plot_model(encoder_model, to_file='vae_mlp_encoder.png', show_shapes=True)
    #    
    latent_inputs = Input(shape=(dim,), name='z_sampling') 
    #--------------decoder (trainable)-----------   
    # block 6
    #decoder = Dense(1000,activation='relu', name = 'dblock_dense1')(latent_inputs)
    decoder = Dense(7*7*512,activation='relu', name = 'dblock_dense2')(latent_inputs)
    decoder = BatchNormalization()(decoder)
    decoder = Reshape([7,7,512],name = 'dblock_reshpe')(decoder)    
    decoder = UpSampling2D((2,2))(decoder)        
       
    # Block 5
    decoder = Conv2D(256, (3, 3),activation='relu', padding='same', name='dblock5_conv1')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = UpSampling2D((2,2))(decoder)
        
    # Block 4
    decoder = Conv2D(128, (3, 3),activation='relu', padding='same', name='dblock3_conv1')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = UpSampling2D((2,2))(decoder)     
             
    # Block 3
    decoder = Conv2D(64, (3, 3),activation='relu', padding='same', name='dblock2_conv1')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = UpSampling2D((2,2))(decoder)        
         
    # Block 2
    decoder = Conv2D(3, (3, 3),activation='relu', padding='same', name='dblock1_conv1')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = UpSampling2D((2,2))(decoder) 
        
    # Block 1
    decoder = Conv2D(3, (3, 3),activation='sigmoid', padding='same', name='dblock1_conv3')(decoder)
    #    instantiate decoder model
    decoder_model = Model(latent_inputs, decoder,name = 'decoder')
    decoder_model.summary()
    #plot_model(decoder_model, to_file='vae_mlp_decoder.png', show_shapes=True)
    
    
    # construct a custom layer to calculate the loss
    class CustomVariationalLayer(Layer):
          
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)
    
        def vae_loss(self, x, z_decoded):
            from tensorflow.python.keras import backend as K 
            x = K.flatten(x)
            z_decoded = K.flatten(z_decoded)
            # Reconstruction loss
            xent_loss = metrics.binary_crossentropy(x, z_decoded)
            # KL divergence
            kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)
    
        # adds the custom loss to the class
        def call(self, inputs):
            x = inputs[0]
            z_decoded = inputs[1]
            loss = self.vae_loss(x, z_decoded)
            self.add_loss(loss, inputs=inputs)
            return x
    
    outputs = decoder_model(encoder_model(input_image)[2])    
    y = CustomVariationalLayer()([input_image, outputs])
    #    outputs = decoder_model(encoder_model(input_image)[2])
    vae = Model(input_image, y)
    vae.compile(optimizer=RMSprop(lr=lr), loss=None)
    vae.summary()
    return encoder_model,decoder_model,vae

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
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
    
    # create model
    encoder_model,decoder_model,vae = build_model_VGG16VAE(args.dimension,args.rate)
    
    if args.weights:
        # load model trained
        print('load weights')
        vae = vae.load_weights(args.weights) 
        print('finish load weights')
        
        # test model
        test_model(encoder_model,decoder_model,x_test)
        # generate new images
        generator_image(decoder_model,args.dimension)
        # tsne
        tsne(encoder_model,x_train)
        
        
    if args.path : 
        #train model
        # split training set and validation set
        x_train, x_valid = train_test_split(x_train, test_size=0.2, random_state=123)

        history = vae.fit(x_train,
                          shuffle=True,
                          epochs=args.epoch,
                          batch_size=32,
                          verbose=1,
                          validation_data=(x_valid,None))
        # save weights of model
        vae.save_weights(args.path)
        #evaluate model
        evaluate_model(history, vae,x_test)
        # test model
        test_model(encoder_model,decoder_model,x_test)
        # generate new images
        generator_image(decoder_model,args.dimension)
        # tsne
        tsne(encoder_model,x_train)
   
    


