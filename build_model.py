# Create Image Data Generator
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.layers import MaxPooling2D, Flatten, Conv2D, Dense
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from sklearn.utils.class_weight import compute_class_weight
import numpy as np



# Define data directories
main_dir = "data/"
train_dir = main_dir + "Train/"
valid_dir = main_dir + "Val/"
test_dir = main_dir + "Prediction/"

# Data Augmentation
img_width, img_height = [224, 224]
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1. /255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode='binary',
                # subset='training',
                shuffle=True)

valid_generator = valid_datagen.flow_from_directory(valid_dir,
                  target_size=(img_width, img_height),
                  # subset='validation',
                  batch_size=batch_size,
                  class_mode='binary')

test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode="binary")

print("train_generator class indices",train_generator.class_indices)

# Build the model
model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', activation='relu', strides=(1, 1), input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(activation='relu', units=128))
model.add(Dense(activation='relu', units=64))
model.add(Dense(activation='sigmoid', units=1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

model_save = ModelCheckpoint('./model/COVID_predictor_model.h5',
                             save_best_only=True,
                             save_weights_only=False,
                             monitor='val_loss',
                             mode='min', verbose=1)

early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.3, min_lr=0.000001)
callbacks_list = [early, learning_rate_reduction, model_save]

# Compute class weights
weights = compute_class_weight(class_weight='balanced',
                               classes=np.unique(train_generator.classes),
                               y=train_generator.classes)

cw = dict(zip(np.unique(train_generator.classes), weights))
print("Class Weights:", cw)

history = model.fit(train_generator, epochs=25, validation_data=valid_generator, class_weight=cw, callbacks=callbacks_list)


# Plot graph between training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Training and Validation Losses')
plt.xlabel('epoch')
plt.savefig('./graphs/training_vs_validation_loss.png')
plt.show()

# Plot graph between training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.xlabel('epoch')
plt.savefig('./graphs/training_vs_validation_accuracy.png')
plt.show()









