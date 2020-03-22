import csv
import os
import shutil

with open('C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\dataset\\train.csv','r') as file:
    reader = csv.reader(file)
    for i in reader:
        if i[0]=="Image":
            continue
        name_folder = str(i[1])
        name_file = str(i[0])
        if name_file=="image11.jpg":
            print(name_folder)
        from_path = "C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\dataset\\Train Images\\%s"%name_file
        to_path = 'C:\\Users\\Nipun\\Documents\\My Projects\\Hackerearth\\%s'%name_folder
        newPath = shutil.copy(from_path, to_path)
        print(i)
