import csv
import os
import shutil
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow import keras

new_model = keras.models.load_model('_model_weights.h5')
predictions = ["Attire","Decorationandsignage","Food","misc"]
output = []
with open('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\dataset\\test.csv') as file:
    reader = csv.reader(file)
    c = 0
    for i in reader:
        c+=1
        if i[0]=="Image":
            continue
        name_file = str(i[0])
        #import image to this
        path = "C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\dataset\\Test Images\\%s"%name_file
        img = image.load_img(path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = new_model.predict(images, batch_size=10)
        maxi = classes[0][0]
        for l in classes[0][1:]:
            if l>maxi:
                maxi = l
        for k in range(4):
            if classes[0][k]==maxi:
                output.append([name_file,predictions[k]])
                print(str(c)+". "+name_file+" | "+predictions[k])
                break
print("output length - ",len(output))
with open('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\dataset\\out_new_9.csv', mode='w', newline='') as csv_file:
    fieldnames = ['Image', 'Class']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for i in output:        
        writer.writerow({'Image': i[0], 'Class': i[1]})