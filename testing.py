import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import *
import tensorflow
from tensorflow.keras.utils import to_categorical
from keras import callbacks
from keras.models import load_model
import os.path

# train_categories = []
# x_train = np.empty((790, 100, 100, 3))
# k = 0
# y_train = []
# for index, i in enumerate(os.listdir("./data/archive/train")):
#     train_categories.append(i)
#     for j in os.listdir('./data/archive/train/'+i):
#         img = Image.open('./data/archive/train/'+i+'/'+j)
#         if img.size[0] > img.size[1]:
#             scale = 100 / img.size[1]
#             new_h = int(img.size[1]*scale)
#             new_w = int(img.size[0]*scale)
#             new_size = (new_w, new_h)
#         else:
#             scale = 100 / img.size[0]
#             new_h = int(img.size[1]*scale)
#             new_w = int(img.size[0]*scale)
#             new_size = (new_w, new_h)

#         resized = img.resize(new_size)
#         resized_img = np.array(resized, dtype=np.uint8)

#         left = 0
#         right = left + 100
#         up = 0
#         down = up + 100
#         cropped = resized.crop((left, up, right, down))
#         cropped_img = np.array(cropped, dtype=np.uint8)
#         if cropped_img.shape[-1] > 3:
#             x_train[k] = x_train[k-1]
#             y_train.append(y_train[-1])
#         else:    
#             x_train[k] = cropped_img
#             y_train.append(index)
#         k += 1


# model = load_model('model.h5')
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# callback = callbacks.EarlyStopping(monitor="loss", patience=3)
# y_train = np.array(y_train)
# y_train = to_categorical(y_train)
# history = model.fit(x=x_train, y=y_train, batch_size=None, callbacks=[callback], epochs=20)

# '''

# # for i in os.listdir("./templates/images"):
# #     train_categories.append(i)

# # model.load_weights("finalmodel.hdf5")
# img = Image.open(newDes)
# original_img = np.array(img, dtype=np.uint8)
# plt.imshow(original_img)
# # plt.show()
# # plt.imshow(cropped_img)

# # plt.show()
# cropped_img = cropped_img / 255
# X = np.reshape(cropped_img, newshape=(1, cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2]))
# plt.imshow(cropped_img)
# plt.show()
# prediction_multi = model.predict(x=X)
# print(prediction_multi)
# # # print(np.argmax(prediction_multi))
# # # print("Fruit is : ", train_categories[np.argmax(prediction_multi)])
# # fruit_name = train_categories[np.argmax(prediction_multi)]

# # acc_sort_index = np.argsort(prediction_multi)
# # top_pred = acc_sort_index[:, -6:]
# # results =[train_categories[top_pred[0][-1]],train_categories[top_pred[0][-2]],train_categories[top_pred[0][-3]]]
# # print(results)
# # print("1st Prediction: ", train_categories[top_pred[0][-1]])
# # print("2nd Prediction: ", train_categories[top_pred[0][-2]])
# # print("3rd Prediction: ", train_categories[top_pred[0][-3]])
# '''


def process_img(path):
    img = Image.open(path)
    if img.size[0] > img.size[1]:
        scale = 100 / img.size[1]
        new_h = int(img.size[1]*scale)
        new_w = int(img.size[0]*scale)
        new_size = (new_w, new_h)
    else:
        scale = 100 / img.size[0]
        new_h = int(img.size[1]*scale)
        new_w = int(img.size[0]*scale)
        new_size = (new_w, new_h)

    resized = img.resize(new_size)
    resized_img = np.array(resized, dtype=np.uint8)

    left = 0
    right = left + 100
    up = 0
    down = up + 100
    cropped = resized.crop((left, up, right, down))
    cropped_img = np.array(cropped, dtype=np.uint8)

    cropped_img = cropped_img / 255

    return cropped_img


# target = os.path.join(APP_ROOT, 'test_images/')
# print(target)

# if not os.path.isdir(target):
#     os.mkdir(target)

# for file in request.files.getlist("file"):
#     print(file)
#     filename = file.filename
#     destination = "/".join([target, filename])
#     print(destination)
#     file.save(destination)	

newDes = os.path.join('test_images/Apple 3.jpg')


train_categories = []
for index, i in enumerate(os.listdir("./data/archive/train")):
    train_categories.append(i)
# train_samples = []
# for i in os.listdir("./data/merged/train"):
#     train_categories.append(i)

# for i in os.listdir("./templates/images"):
#     train_categories.append(i)

# model.load_weights("finalmodel.hdf5")
# img = Image.open(newDes)
# original_img = np.array(img, dtype=np.uint8)
# # plt.imshow(original_img)

# if img.size[0] > img.size[1]:
#     scale = 100 / img.size[1]
#     new_h = int(img.size[1]*scale)
#     new_w = int(img.size[0]*scale)
#     new_size = (new_w, new_h)
# else:
#     scale = 100 / img.size[0]
#     new_h = int(img.size[1]*scale)
#     new_w = int(img.size[0]*scale)
#     new_size = (new_w, new_h)

# resized = img.resize(new_size)
# resized_img = np.array(resized, dtype=np.uint8)
# # plt.imshow(resized_img)

# left = 0
# right = left + 100
# up = 0
# down = up + 100

# cropped = resized.crop((left, up, right, down))
# cropped_img = np.array(cropped, dtype=np.uint8)
# plt.imshow(cropped_img)


model = load_model('82acc_model.h5')
cropped_img = process_img(newDes)
X = np.reshape(cropped_img, newshape=(1, cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2]))
prediction_multi = model.predict(x=X)
print(prediction_multi)
print("hello")
print(np.argmax(prediction_multi))

print("Fruit is : ", train_categories[np.argmax(prediction_multi)])

    # fruit_name = train_categories[np.argmax(prediction_multi)]