__author__ = "https://www.linkedin.com/in/bongsang/"
__license__ = "MIT"

from utils import download
from utils import dataset

import os
from os.path import join
import zipfile
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import Callback


# Define download directories
download_path = "download"
cat_download_path = join(download_path, "PetImages", "Cat")
dog_download_path = join(download_path, "PetImages", "Dog")

# Define dataset directories
dataset_path = "dataset"
cat_train_path = join(dataset_path, "train", "cat")
dog_train_path = join(dataset_path, "train", "dog")
cat_validation_path = join(dataset_path, "validation", "cat")
dog_validation_path = join(dataset_path, "validation", "dog")


def dataset_setup(url):
    ## Download from URL
    if not os.path.exists(download_path):
        filename = "cats-and-dogs.zip"
        file = download.from_url(url, path=download_path, file=filename)
        zfile = zipfile.ZipFile(file, 'r')
        zfile.extractall(download_path)
        zfile.close()

    ## Setup dataset from downloaded data
    if not os.path.exists(dataset_path):
        os.makedirs(cat_train_path)
        os.makedirs(dog_train_path)
        os.makedirs(cat_validation_path)
        os.makedirs(dog_validation_path)

        ### Splitting dataset
        split_rate = .9  # Training: 90%, Testing: 10%
        dataset.split(cat_download_path, cat_train_path, cat_validation_path, split_rate)
        dataset.split(dog_download_path, dog_train_path, dog_validation_path, split_rate)

    num_train = len(os.listdir(cat_train_path))
    num_test = len(os.listdir(cat_validation_path))
    print(f"The total number of Dataset is {num_train + num_test} (Cats: {num_train}, Dogs: {num_test})")


def data_generator(augmentation=False):
    if augmentation:
        # Reducing over-fitting by various augmentation
        train_data_generator = image.ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        valid_data_generator = image.ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    else:
        train_data_generator = image.ImageDataGenerator(rescale=1.0 / 255)
        valid_data_generator = image.ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_data_generator.flow_from_directory(
        join(dataset_path, "train"),
        batch_size=20,
        class_mode="binary",
        target_size=(150, 150))

    validation_generator = valid_data_generator.flow_from_directory(
        join(dataset_path, "validation"),
        batch_size=20,
        class_mode="binary",
        target_size=(150, 150)
    )

    return train_generator, validation_generator


def model_design():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model


class AccCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        elif logs.get('acc') > 0.9:
            self.model.stop_training = True
            print(f"\nEarly stopping! Epoch: {epoch} \t Accuracy: {round(logs.get('acc') * 100)}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="ex) python main.py --mode=train --epochs=100 --url=\"http://...\"")
    parser.add_argument('--mode', default='train', choices=['train', 'test'], required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--url', default="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip")
    parser.add_argument('--test_path', default='tests')
    parser.add_argument('--result_path', default='results')
    args = parser.parse_args()

    # Tensorflow GPU setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # ----------------------------------------
    # Download data from URL and setup dataset
    # ----------------------------------------
    dataset_setup(args.url)

    # ---------------
    # model designing
    # ---------------
    model = model_design()
    model.compile(
        optimizer=RMSprop(lr=1e-4),
        loss=BinaryCrossentropy(),
        metrics=['acc'])

    model.summary()

    # -------------
    # model fitting
    # -------------
    print(f"args.mode={args.mode}")

    fig_path = join(".", args.result_path)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if 'train' in args.mode:
        print("###### Model training ######")
        callback = AccCallback()
        train_generator, validation_generator = data_generator(augmentation=True)

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=args.epochs,
            validation_data=validation_generator,
            validation_steps=50,
            callbacks=[callback],
            verbose=2
        )

        # ----------------------
        # Train history plotting
        # ----------------------
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))
        plt.plot(epochs, acc, 'ro', label='Training accuracy')
        plt.plot(epochs, val_acc, 'bo', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(join(fig_path, "accuracy.jpg"))

        plt.figure()
        plt.plot(epochs, loss, 'ro', label='Training Loss')
        plt.plot(epochs, val_loss, 'bo', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(join(fig_path, "loss.jpg"))
    else:
        print("###### Model testing ######")
        test_path = join(".", args.test_path)
        test_files = dataset.test(test_path)

        for file in test_files:
            test_image = image.load_img(join(test_path, file), target_size=(150, 150))
            x = image.img_to_array(test_image)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)

            if classes[0] > 0.5:
                predition = "Prediction = Dog!"
                print(f"{file} is a dog.")
            else:
                predition = "Prediction = Cat!"
                print(f"{file} is a cat.")

            plt.imshow(test_image)
            plt.title(predition)
            plt.savefig(join(fig_path, "prediction_"+file))



