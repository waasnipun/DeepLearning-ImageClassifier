from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.optimizer_v2.adam import Adam
from matplotlib import pyplot as plt
import os

HEIGHT = 64
WIDTH = 64

base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape=(HEIGHT, WIDTH, 3))



TRAIN_DIR = 'C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images'
HEIGHT = 64
WIDTH = 64
BATCH_SIZE = 8

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
      rescale = 1./255,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                    target_size=(HEIGHT, WIDTH), 
                                                    batch_size=BATCH_SIZE)


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

FC_LAYERS = [1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(base_model, 
                                      dropout=dropout, 
                                      fc_layers=FC_LAYERS, 
                                      num_classes=4)
train_Attire = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images\\Attire')
train_Decorationandsignage = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images\\Decorationandsignage')
train_Food = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images\\Food')
train_misc = os.path.join('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\train_images\\misc')


class_weights = {}
predictions = ["Attire","Decorationandsignage","Food","misc"]
weight =  [len(os.listdir(train_Attire)),len(os.listdir(train_Decorationandsignage)),len(os.listdir(train_Food)),len(os.listdir(train_misc))]
c = 0
for i in predictions:
    class_weights[c] = weight[c]
    c+=1
print(class_weights)

NUM_EPOCHS = 60
BATCH_SIZE = 8
num_train_images = 5000

adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

filepath= 'C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\'+ 'ResNet50\\' + '_model_weights.h5'
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]

history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8, 
                                       steps_per_epoch=num_train_images // BATCH_SIZE, 
                                       shuffle=True, callbacks=callbacks_list,class_weight=class_weights)




# Plot the training and validation loss + accuracy
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs.png')
plot_training(history)