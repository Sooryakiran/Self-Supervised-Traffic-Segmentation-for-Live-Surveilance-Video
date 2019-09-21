import numpy as np
import cv2
import random

class Memory:

    def __init__(self, max_size, image_size = [500, 500, 3], batch_size = 8):

        """
        The class constructor

        """
        self.max_size = max_size

        if len(image_size) != 3:
            raise ValueError("Image size should be a list [Width, Height, Channels]")
        else:
            self.image_size = image_size
        self.batch_size = batch_size
        self.__data_x = []
        self.__data_y = []

    def add(self, image, target):

        """
        Adds a new image to the memory

        """

        if np.asarray(image.shape).all() != np.asarray(self.image_size).all():
            raise ValueError("Image size is not compatible")

        self.__data_x.append(image)
        self.__data_y.append(target)
        if len(self.__data_x) > self.max_size:
            self.__data_x.pop(0)
            self.__data_y.pop(0)

        return 1

    def get_batch(self):
        """
        Randomly samples training images and targets from the memory

        """
        length = len(self.__data_x)
        if length <= 0:
            raise ValueError("Memory buffer is empty")
        random_vector = [random.randint(0, length - 1) for _ in range(self.batch_size)]
        train_x = []
        train_y = []

        for i in random_vector:
            train_x.append(self.__data_x[i])
            train_y.append(self.__data_y[i])

        return np.asarray(train_x), np.asarray(train_y)

if __name__ == "__main__":
    print("This is not an executable file")
    print("Please run main.py")
