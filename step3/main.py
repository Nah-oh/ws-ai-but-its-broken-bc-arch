import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

NUMBER_TRY = 14

TRAINING_DIR = "../rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


VALIDATION_DIR = "../rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
    batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
    batch_size=126
)

model = tf.keras.models.Sequential([
    # the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(input_shape=(150,150,3),filters=64,kernel_size=(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    # The second convolution
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    # The third convolution
    tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    # The fourth convolution
    tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(data_format=None),
    # 512 neuron hidden layer
    tf.keras.layers.Dropout(rate=0.5),
])

#compile model with loss as categorical_crossentropy, optimizer as rmsprop and metrics as accuracy
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#train the model and test his validity
history = model.fit(train_generator, epochs=NUMBER_TRY, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("my_model.h5")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
