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
prior = keras.applications.VGG16(
    include_top=False, 
    weights='imagenet',
    input_shape=(48, 48, 3)
)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 3)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model = Sequential()
model.add(prior)
model.add(Flatten())
model.add(Dense(256, activation='relu', name='Dense_Intermediate'))
model.add(Dropout(0.1, name='Dropout_Regularization'))
model.add(Dense(4, activation='sigmoid', name='Output'))
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='softmax'))
# model.summary()


# for cnn_block_layer in model.layers[0].layers:
#     cnn_block_layer.trainable = False
# model.layers[0].trainable = False

class_weights = {}
predictions = ["Attire","Decorationandsignage","Food","misc"]
weight =  [len(os.listdir(train_Attire)),len(os.listdir(train_Decorationandsignage)),len(os.listdir(train_Food)),len(os.listdir(train_misc))]
c = 0
for i in predictions:
    class_weights[c] = weight[c]
    c+=1



# model.compile(loss='categorical_crossentropy',
#               optimizer=tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6),
#               metrics=['accuracy'])

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
    target_size=(48, 48),
    class_mode='categorical',
    subset='training',
)
validation_generator = train_datagen.flow_from_directory(
    image_directory_validate,
    target_size=(48, 48),
    class_mode='categorical',
    subset='validation'
)
# Compile the model. I found that RMSprop with the default learning
# weight worked fine.
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames) // batch_size,
    epochs=150,
    validation_data=validation_generator,
    validation_steps=len(train_generator.filenames) // batch_size,
    class_weight=class_weights,
    # callbacks=[
    #     EarlyStopping(patience=3, restore_best_weights=True),
    #     ReduceLROnPlateau(patience=2)
    # ]
)

scores = model.evaluate(validation_generator)

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
model.save("Trained_model_1.h5")


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

