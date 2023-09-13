import os
import random

from keras import preprocessing

import numpy as np


def encode_answer(label):

        encoded_label = np.zeros((5, 62), dtype=np.uint8)

        for i in range(5):
            index = ord(label[i])

            if 48 <= index <= 57:
                index -= 48

            if 97 <= index <= 122:
                index -= 97 - 10

            if 65 <= index <= 90:
                index -= 65 - 36

            encoded_label[i][index] = 1

        return encoded_label


def decode_answer(encoded_label):

    decoded_label = ''

    for i in range(5):

        index = np.argmax(encoded_label[i])

        character = ''

        if 0 <= index <= 9:
            character = chr(index + 48)

        if 10 <= index <= 35:
            character = chr(index + 97 - 10)

        if 36 <= index <= 61:
            character = chr(index + 65 - 36)

        decoded_label += character

    return decoded_label


class CaptchaSolverModel:

    _DATASET_PATH = 'data'

    def __init__(self):
        self.training_data, self.testing_data = self.split_data()

    def load_data(self):
        data = {}

        image_count = 0

        for training_image in os.listdir(self._DATASET_PATH):

            label = training_image.split('.')[0]

            data[label] = preprocessing.image.load_img(os.path.join(self._DATASET_PATH, training_image))

            image_count += 1

            print("Loaded Image: {}, Count: {}".format(training_image, image_count))

        return data

    def split_data(self):

        training_data = {}
        testing_data = {}

        for label, image in self.load_data().items():

            if random.randint(0, 9) < 2:
                testing_data[label] = image
            else:
                training_data[label] = image

        print("Training Data: {}, Testing Data: {}".format(len(training_data), len(testing_data)))

        return training_data, testing_data