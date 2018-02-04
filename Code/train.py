import pickle
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# # config.gpu_options.allow_growth = True
# # session = tf.Session(config=config)
# set_session(tf.Session(config=config))

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.optimizers import Adam,RMSprop
from keras.callbacks import ModelCheckpoint

X_train, Y_train, X_test, Y_test = pickle.load(open("../Data/full_dataset.pkl","rb"), encoding='latin')

datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True, rotation_range=25,
                             zoom_range=0.2)
datagen.fit(X_train)

data_generator = datagen.flow(X_train, Y_train, batch_size=96)

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(32,32,3), activation='relu'))
#new
model.add(Conv2D(128, (3,3), activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

#new
model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

cb = ModelCheckpoint("../Data/model/model.h5", monitor='val_loss', save_best_only=True)
model.fit_generator(data_generator, epochs=30, shuffle=True, validation_data=(X_test, Y_test), verbose=2, callbacks=[cb]
                    # , steps_per_epoch=3
                    )

model.save_weights('../Data/weights/cnn.h5')
model.save('../Data/model/model.h5')
print("Saved")







