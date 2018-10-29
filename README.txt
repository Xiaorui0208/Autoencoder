AUTOENCODER

AutoEncoder - Keras implementation on ImageNet and synthetic dataset

Dependencies

keras(TensorFlow backend)
numpy, matplotlib, scipy, glob,cv2
cuda 9.0 , cudnn 7.0.5 

Usage
1)  Manage dataset
You need to do some preprocessing on dataset. These images are cropped to 224*224*3 pixels as the input images.
You can run

src/prepocess.py :
it  crops the input images and resizes them
example : python3 prepocess.py  -p ‘dbcafe/x_train/’  - d  224
            -p means the path of dataset, -d means the dimension to crop
2) Train and Test your model
There are 5 models, you can choose one of them to train and test.
You can run 

src/AE_deep_spatial.py :
it creates, trains and test deep spatial autoencoder .
After training, you can save your model in save_model directory, and then you can test your model, the result can be saved in result_cafe directory.
example:  python3 AE_deep_spatial.py

src/ ae.py :
it creates, trains and test VGG-16  autoencoder .
After training, you can save your model in save_model directory, and then you can test your model, the result can be saved in result_cafe/AE/ directory.
example:  python3  ae.py  -d 32 -r 0.0001 -e 50 - p ‘save_model/ae.h5’
 -d means the dimension of latent space, -r means learning rate - e means training epochs, -p means the path for saving model.
if you use the model trained and test the model, you can run
        python3  ae.py  -m ‘save_model/ae.h5’
 -m means the path of model trained

src/ ae_sparse.py :
it creates, trains and test sparse VGG-16  autoencoder .
After training, you can save your model in save_model directory, and then you can test your model, the result can be saved in result_cafe/AE_SPARSE/ directory.
example:  python3  ae_sparse.py  -d 32 -r 0.0001 -e 50 -re 15e-7
                   - p ‘save_model/ae_sparse.h5’
 -d means the dimension of latent space, -r means learning rate - e means training epochs, -re means the regularizer parameter -p means the path for saving model.
if you use the model trained and test the model, you can run
        python3  ae_sparse.py  -m ‘save_model/ae_sparse.h5’
 -m means the path of model trained

src/ vae.py :
it creates, trains and test variational VGG-16  autoencoder .
After training, you can save the weights of model in save_model directory, the saved model can be used to analyse the latent distribution and to generate new images, and then you can test your model, the result can be saved in result_cafe/VAE/ directory.
example:  python3  vae.py  -d 32 -r 0.0001 -e 50 - p ‘save_model/vae.h5’
 -d means the dimension of latent space, -r means learning rate - e means training epochs, -p means the path for saving the weights of model.
if you load the weights of model and test the model, you can run
        python3  vae.py  -m ‘save_model/vae.h5’
 -m means the path of loading the weights of model.

src/ vae_sparse.py :
it creates, trains and test sparse and variational VGG-16  autoencoder .
After training, you can save the weights of model in save_model directory, the saved model can be used to analyse the latent distribution and to generate new images,and then you can test your model, the result can be saved in result_cafe/VAE_SPARSE directory.
example:  python3  vae-sparse.py  -d 32 -r 0.0001 -e 50 -re 15e-7
                    - p ‘save_model/vae_sparse.h5’
 -d means the dimension of latent space, -r means learning rate - e means training epochs,  -re means the regularizer parameter, -p means the path for saving the weights of model.
if you load the weights of model and test the model, you can run
        python3  vae_sparse.py  -m ‘save_model/vae_sparse.h5’
 -m means the path of loading the weights of model.

3) Analysis
If you want to analyze the latent space for sparse VGG-16 autoencoder, you can run
src/ sparse_decoder.py :
it analyzes the latent space by changing four nodes and linear transformation.
example:  python3  sparse_decoder.py   0   -p '.dbcafe/cardboard/' 
             0 means that you choose the method of changing four nodes, -p means 
              the path of dataset, you can save the results in result-cafe/generator/ directory
example:  python3  sparse_decoder.py  1
              1 means that you choose the method of linear transformation, you can save
               the results in result-cafe/linear/ directory
 



Directory
/home/xhuo/AE/
- src(code)
   -- AE_deep_spatial.py
   -- ae_sparse.py
   -- sparse_decoder.py
   -- vae_sparse.py
   -- ae.py
   -- preprocess.py
   -- vae.py
- data(small synthetic dataset)
   -- x_train(training set)
   -- x_test(test set)
- dbcafe(big synthetic dataset)
   -- x_train(training set)
   -- x_test(test set)
   -- two(used for linear transformation)
   -- box(one kind of object used for changing four nodes)
   -- cardboard
   -- can
   --  cinder
   -- mailbox
- result(the testing results of different models on small synthetic dataset)
   -- AE
   -- VAE
   -- AE_SPARSE
- result_cafe(the testing results of different models big synthetic dataset)
   -- AE
   -- VAE
   -- AE_SPARSE
   -- VAE_SPARSE
   -- linear (results for linear transformation)
   -- generator(results for changing four nodes)
- save_model(saving model)

