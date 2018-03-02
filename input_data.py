import os
import cv2
import random
import numpy as np

DATA_DIR = 'data/imagesPlaces205_resize/'
INDEX_FILE_TRAIN = 'data/Train_places205_Random_Suffle.csv'
INDEX_FILE_TEST = 'data/val_places205.csv'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def get_image_for_train(image_path):
    """
    Read image from image_path and performs 
    some random operations on it before returning
    ARGS:
        image_path : full image path and name
    OUTPUT:
        Colored L*a*b* image
        Pencil Sketch L*a*b* image

    Note: range of L,a,b is (0,255)
    """
    image_color = cv2.imread(image_path)
    original_height, original_width, _ = image_color.shape
    
    # Randomly crop a [height, width] section of the image
    y = random.randint(0,original_height-IMAGE_HEIGHT)
    x = random.randint(0,original_width-IMAGE_WIDTH)
    distorted_image_color = image_color[y:y+IMAGE_HEIGHT, x:x+IMAGE_WIDTH]

    # Randomly flip the image horizontally
    if random.randint(0,1) == 1:
        distorted_image_color = cv2.flip(distorted_image_color, 1)

    # Convert into Gray Scale
    image_gray = cv2.cvtColor(distorted_image_color, cv2.COLOR_BGR2GRAY)

    # Convert to Pencil Sketch
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_blur = cv2.GaussianBlur(255-image_gray, (21,21), 0, 0)
    img_blend = cv2.divide(image_gray, 255-img_blur, scale=256)
    img_blend = cv2.multiply(img_blend, 255-img_blur, scale=1./256)
    pencil_sketch = clahe.apply(img_blend)

    # Converting the pencil_sketch into dimentions [IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS=1]
    pencil_sketch = np.expand_dims(pencil_sketch, axis=2)
    
    # Convert to L*a*b* mode
    lab_img_color = cv2.cvtColor(distorted_image_color, cv2.COLOR_BGR2LAB)
    
    
    return lab_img_color, pencil_sketch


def get_batches_for_train(batch_size):
    """
    Generate batches of batch size for training
    ARGS:
        batch_size : An integer
    OUTPUT:
        Batch of Colored images
        Batch of Pencil Sketch images
        Batch of Labels

    Note: Images are in the L*a*b* color encoding
    """
    with open(INDEX_FILE_TRAIN) as f:
        images_color = []
        images_pencil = []
        labels = []
        i=1
        for line in f:
            _,filename,label = line.split(',')
            label = int(label.partition('\n')[0])
            img_color, img_pencil = get_image_for_train(DATA_DIR + filename)
            images_color.append(img_color)
            images_pencil.append(img_pencil)
            labels.append([label])
            i+=1
            if i>batch_size:
                i=1
                yield images_color,images_pencil,labels
                images_color=[]
                images_pencil=[]
                labels=[]

def get_image_for_test(image_path):
    """
    Read image from image_path and resizes according to given size
    ARGS:
        image_path : full image path and name
    OUTPUT:
        Colored L*a*b* image
        Pencil Sketch L*a*b* image

    Note: range of L,a,b is (0,255)
    """
    image_color = cv2.imread(image_path)
    original_height, original_width, _ = image_color.shape
    
    # Resize to image size
    image_color = cv2.resize(image_color, (IMAGE_WIDTH,IMAGE_HEIGHT))
    
    # Convert into Gray Scale
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # Convert to Pencil Sketch
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_blur = cv2.GaussianBlur(255-image_gray, (21,21), 0, 0)
    img_blend = cv2.divide(img_gray, 255-img_blur, scale=256)
    img_blend = cv2.multiply(img_blend, 255-img_blur, scale=1./256)
    pencil_sketch = clahe.apply(img_blend)

    # Converting the pencil_sketch into dimentions [IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS=1]
    pencil_sketch = np.expand_dims(pencil_sketch, axis=2)
    
    # Convert to L*a*b* mode
    lab_img_color = cv2.cvtColor(distorted_image_color, cv2.COLOR_BGR2LAB)
    
    
    return lab_img_color, pencil_sketch


def get_batches_for_test(batch_size):
    """
    Generate batches of batch size for testing
    ARGS:
        batch_size : An integer
    OUTPUT:
        Batch of Colored images
        Batch of Pencil Sketch images
        Batch of Labels

    Note: Images are in the L*a*b* color encoding
    """
    with open(INDEX_FILE_TEST) as f:
        images_color = []
        images_pencil = []
        labels = []
        i=1
        for line in f:
            filename,label = line.split(' ')
            label = int(label.partition('\n')[0])
            img_color, img_pencil = get_image_for_test(DATA_DIR + filename)
            images_color.append(img_color)
            images_pencil.append(img_pencil)
            labels.append([label])
            i+=1
            if i>batch_size:
                i=1
                yield images_color,images_pencil,labels
                images_color=[]
                images_pencil=[]
                labels=[]

def get_real_pencil_sketch(image_path):
    """
    Creates input for testing real human made pencil sketches
    ARGS:
        image_path : full image path and name
    OUTPUT:
        Pencil Sketch image with L*a*b* color mode
    """
    # Loading file
    image_gray = cv2.imread(image_path,0)

    # Resizing it
    resized_image_gray = cv2.resize(image_gray, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Converting to input required for network (real pencil sketch/ enhancing it)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(1,1))
    pencil_sketch = clahe.apply(resized_image_gray)

    return pencil_sketch
