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
from keras.engine.saving import load_model

train_Attire = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images\\Attire')
train_Decorationandsignage = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images\\Decorationandsignage')
train_Food = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images\\Food')
train_misc = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images\\misc')

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(192, 192, 3), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# load the pretrained model
prior = load_model('Trained_model_new_2.h5')

# add all but the first two layers of VGG16 to the new model
# strip the input layer out, this is now 96x96
# also strip out the first convolutional layer, this took the 96x96 input and convolved it but
# this is now the job of the three new layers.
for layer in prior.layers[1:]:
    layer.name += '_prior'  # set layer names to avoid name collisions
    model.add(layer)

# the pretrained CNN layers are already marked non-trainable
# mark off the top layers as well
for layer in model.layers[-4:]:
    layer.trainable = False
    
# set layer names (otherwise names may collide)
for i, layer in enumerate(model.layers):
    layer.name = 'layer_' + str(i)
# compile the model
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

class_weights = {}
predictions = ["Attire","Decorationandsignage","Food","misc"]
weight =  [len(os.listdir(train_Attire)),len(os.listdir(train_Decorationandsignage)),len(os.listdir(train_Food)),len(os.listdir(train_misc))]
c = 0
for i in predictions:
    class_weights[c] = weight[c]
    c+=1

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
validation_datagen = ImageDataGenerator(
    rescale=1./255,
)

batch_size = 128

image_directory_train = 'C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images'
image_directory_validate = 'C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\validate_images'

train_generator = train_datagen.flow_from_directory(
    image_directory_train,
    target_size=(192, 192),
    class_mode='categorical',
    subset='training',
)
validation_generator = train_datagen.flow_from_directory(
    image_directory_validate,
    target_size=(192, 192),
    class_mode='categorical',
    subset='validation'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames) // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=len(train_generator.filenames) // batch_size,
    class_weight=class_weights,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(patience=2)
    ]
)

scores = model.evaluate(validation_generator)

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
model.save("Trained_model_new_3.h5")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy (transfer learning step 1)')
plt.legend(loc=0)

plt.show()
