from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
from keras import models
from keras import Model
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio as im
import numpy as np


training_images_path = '../car_dataset/data/training'
validation_images_path = '../car_dataset/data/validation'

# Defining ImageDataGenerator Class for training
train_datagen = ImageDataGenerator(
    rescale=1/255,  # normalization
    # data augmentation
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'  # this is default
)

# Defining ImageDataGenerator Class for validation
validation_generator = ImageDataGenerator(rescale=1/255)


# Loading training data using path
train_generator = train_datagen.flow_from_directory(
    training_images_path,
    target_size=(150, 150),
    batch_size=40,
    class_mode='binary'
)


# Loading validation data using path
validation_generator = validation_generator.flow_from_directory(
    validation_images_path,
    target_size=(150, 150),
    batch_size=40,
    class_mode='binary'
)

model = models.Sequential([
    # First Convolution
    layers.Conv2D(
        16, (3, 3), activation='relu', input_shape=(150, 150, 3)
    ),
    layers.MaxPooling2D(2, 2),
    # Second Convolution
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    # Third Convolution
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    # Flatten
    layers.Flatten(),
    # Dense layer
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['accuracy']
)

checkpointer = ModelCheckpoint(filepath="best_weights.hd5",
                               monitor='val_accuracy',
                               verbose=1,
                               save_best_only=True)

history = model.fit(
    train_generator,
    steps_per_epoch=200,
    epochs=17,
    callbacks=[checkpointer],
    verbose=1,
    validation_data=validation_generator
)

# Getting the accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plotting the accuracy
epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
# Plotting the loss
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Creating image tensor
# img = image.load_img(validation_images_path + '/damage/2010.jpg',
#                      target_size=(150, 150))
# img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img_tensor, axis=0)
# img_tensor /= 255.


# Instantiating a model from an input tensor and a list of output tensors
# layer_outputs = [layer.output for layer in model.layers[:6]]  # First 6layers
# # Creates a model that will return these outputs, given the model input
# activation_model = Model(
#     inputs=model.input, outputs=layer_outputs)

# # Running the model in predict mode
# # Returns a list of Numpy arrays: one array per layer activation
# activations = activation_model.predict(img_tensor)

# # Visualizing every channel in every intermediate activation
# layer_names = []
# for layer in model.layers[:6]:
#     # Name of the layers, can be part of plot
#     layer_names.append(layer.name)

# images_per_row = 8

# # prints 6 activations
# # print('Length: ' + str(len(activations)))

# # Displays the feature map
# for layer_name, layer_activation in zip(layer_names, activations):
#     # Number of features in the feature map
#     n_features = layer_activation.shape[-1]
#     # The feature map has shape (1, size, size, n_features)
#     size = layer_activation.shape[1]
#     n_cols = n_features // images_per_row
#     display_grid = np.zeros((size * n_cols, images_per_row * size))
#     for col in range(n_cols):  # Tiles each filter into a big horizontal grid
#         for row in range(images_per_row):
#             channel_image = layer_activation[0,
#                                              :, :,
#                                              col * images_per_row + row]
#             # Post-processes the feature to make it visually palatable
#             #  channel_image -= channel_image.mean()
#             #  channel_image = channel_image / channel_image.std()
#             #  channel_image *= 64
#             #  channel_image += 128
#             #  channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#             display_grid[col * size: (col + 1) * size,  # Displays the grid
#                          row * size: (row + 1) * size] = channel_image

#     scale = 1. / size
#     plt.figure(figsize=(scale * display_grid.shape[1],
#                         scale * display_grid.shape[0]))
#     plt.title(layer_name)
#     plt.imshow(display_grid, aspect='auto', cmap='viridis')
#     plt.show()
