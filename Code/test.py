import numpy as np
from keras.models import load_model
import scipy

model = load_model("../Data/model/model.h5" )
DIR = "../Test_Images/"

import os
for each_file in os.listdir(DIR):
    img = scipy.ndimage.imread(DIR+each_file, mode="RGB")
    img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
    a = np.array(img)
    a = a.reshape(1,32,32,3)
    # print(a.shape)
    prediction = model.predict(a)
    is_bird = np.argmax(prediction[0]) == 1
    # print(prediction)
    if is_bird:
        print(each_file," is a bird!")
    else:
        print(each_file,"is not a bird!")