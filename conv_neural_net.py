import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf 
import cv2 as cv
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

#for trainign data
train_Attire = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images\\Attire')
train_Decorationandsignage = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images\\Decorationandsignage')
train_Food = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images\\Food')
train_misc = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images\\misc')

#names of the files
train_Attire_names = os.listdir(train_Attire)
train_Decorationandsignage_names = os.listdir(train_Decorationandsignage)
train_Food_names = os.listdir(train_Food)
train_misc_names = os.listdir(train_misc)

#number of images in the directories
print('total training Attire images:', len(os.listdir(train_Attire)))
print('total training Decorationandsignage images:', len(os.listdir(train_Decorationandsignage)))
print('total training Food images:', len(os.listdir(train_Food)))
print('total training misc images:', len(os.listdir(train_misc)))
print("\n")

#for valdating data
validate_Attire = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\validate_images\\Attire')
validate_Decorationandsignage = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\validate_images\\Decorationandsignage')
validate_Food = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\validate_images\\Food')
validate_misc = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\validate_images\\misc')

#names of the files
validate_Attire_names = os.listdir(validate_Attire)
validate_Decorationandsignage_names = os.listdir(validate_Decorationandsignage)
validate_Food_names = os.listdir(validate_Food)
validate_misc_names = os.listdir(validate_misc)

#number of images in the directories
print('total validate Attire images:', len(os.listdir(validate_Attire)))
print('total validate Decorationandsignage images:', len(os.listdir(validate_Decorationandsignage)))
print('total validate Food images:', len(os.listdir(validate_Food)))
print('total validate misc images:', len(os.listdir(validate_misc)))
print('\n')

#----------------------model---------------------------
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


validation_datagen = ImageDataGenerator(
    rescale=1./255,
)

batch_size = 128

image_directory_train = 'C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images'
image_directory_validate = 'C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\validate_images'

train_generator = training_datagen.flow_from_directory(
	image_directory_train,
	target_size=(64,64),
	class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
	image_directory_validate,
	target_size=(64,64),
	class_mode='categorical'
)
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])



model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit_generator(train_generator, epochs=50, validation_data = validation_generator, verbose = 1)

scores = model.evaluate(validation_generator)

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
model.save("Trained_model_correct_1_3.h5")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)

plt.show()

