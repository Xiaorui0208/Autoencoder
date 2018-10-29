#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 16:05:58 2018

@author: xhuo
"""


from PIL import Image
import glob


# prepocessing on image, images are cropped
def image_transfer(path,dimension):
    """
    Arguments:
        path (string): the path of images
        dimension (int): the dimension of image
    """
    
    files = glob.glob(path + '*png')

    for f in files:
        # convert the image to RGB mode 
        Image.open(f).convert('RGB').save(f)   
        # read image size
        im = Image.open(f) 
        (x, y) = im.size 
        # define standard width
        x_s = dimension 
        # calculate height based on standard width
        y_s = dimension 
        # resize image with high-quality
        out = im.resize((x_s, y_s), Image.ANTIALIAS)  
        out.save(f)
    print('------transfer image finished-------')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    help_ = "the path of dataset"
    parser.add_argument("-p","--path", help=help_)
    help_ = "the dimension of limage"
    parser.add_argument("-d", "--dimension",type=int, default=224, help=help_)
    args = parser.parse_args()
    
    if args.path:
        image_transfer(args.path,args.dimension)
    else :
        print("input the path of dataset")
        