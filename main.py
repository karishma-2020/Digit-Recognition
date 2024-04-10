# import libraries
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2  
import matplotlib.pyplot as plt
from tensorflow.python.keras.metrics import accuracy


# Dataset
mnist = tf.keras.datasets.mnist

#preprocessing functions

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image

def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def apply_filter(image, filter_type):
    if filter_type == 'blur':
        filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == 'edge':
        filtered_image = cv2.Canny(image, 100, 200)
    elif filter_type == 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        filtered_image = cv2.filter2D(image, -1, kernel)
    else:
        print("Invalid filter type.")
        return None
    return filtered_image

def model_training():

    #model training
    (x_train, y_train),(x_test,y_test)=mnist.load_data()# split the data in training set as tuple
    x_train = tf.keras.utils.normalize(x_train , axis = 1)
    x_test = tf.keras.utils.normalize(x_test , axis = 1)
    print(x_train)

    model= tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))
    model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train,y_train, epochs=4)#As the number of epochs increases beyond 11,chance of overfitting of the model on training data

    loss , accuracy  =model.evaluate(x_test,y_test)
    print(accuracy)
    print(loss)
    return model

    


def preprocessing(image_path):
    # Load the image
    # image_path = '1.png'
    original_image = cv2.imread(image_path)

    # Resize the image
    scaled_image = resize_image(original_image, scale_percent=50)

    # Convert the image to grayscale
    grayscale_image = convert_to_grayscale(scaled_image)

    # Apply filters
    blurred_image = apply_filter(grayscale_image, filter_type='blur')
    edge_detected_image = apply_filter(grayscale_image, filter_type='edge')
    sharpened_image = apply_filter(grayscale_image, filter_type='sharpen')

    # Save the processed images
    
    plt.title("Blurred Image")
    plt.imshow(blurred_image,cmap=plt.cm.binary)
    plt.show()
    
    plt.title("Edge Detection")
    plt.imshow(edge_detected_image,cmap=plt.cm.binary)
    plt.show()

    plt.title("Sharpened Image")
    plt.imshow(sharpened_image,cmap=plt.cm.binary)
    plt.show()
    
    

    print("Image processing complete. Processed images saved successfully.")

    

def main():


    model=model_training()

    

    for x in range(1,5):
    # now we are going to read images it with open cv

        img=cv2.imread(f'{x}.png')[:,:,0]#all of it and 1st and last one
       

        img=np.invert(np.array([img]))#invert black to white in images so that model wont get confues
        prediction=model.predict(img)
        print("----------------")
        print("The predicted value is : ",np.argmax(prediction))
        print("----------------")
        plt.title("Input Image")
        plt.imshow(img[0],cmap=plt.cm.binary)#change the color in black and white
        plt.show()
        preprocessing(str(x) + ".png")


if __name__ == "__main__":
    main()
