import csv
import time
import os

# load the labels
y_train = []
with open("data/train/y_train.csv",'r') as f:
    data = csv.reader(f, delimiter=',')
    for i in data:
        if i[0] == '10':
            y_train.append(0)
        else:
            y_train.append(int(i[0]))
y_test = []
with open("data/test/y_test.csv",'r') as f:
    data = csv.reader(f, delimiter=',')
    for i in data:
        if i[0] == '10':
            y_test.append(0)
        else:
            y_test.append(int(i[0]))
y_extra = []
with open("data/extra/y_extra.csv",'r') as f:
    data = csv.reader(f, delimiter=',')
    for i in data:
        if i[0] == '10':
            y_extra.append(0)
        else:
            y_extra.append(int(i[0]))

# sort the pictures
for i in range(1, len(y_extra)+1):
    os.rename(f"data/extra/images/extra_dataset_{i}.png", f"sorted_data/extra/{y_extra[i-1]}/extra{i}.png")
    print(f"Moved {i} files!")

print("Done!")





