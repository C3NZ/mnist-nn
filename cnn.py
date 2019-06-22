from keras.datasets import mnist
from keras.layers import Conv2D, Dense, MaxPooling2D
from keras.models import Sequential


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))


def obtain_model_data():
    (X_train, X_test), (y_train, y_test) = mnist.load_data()


def main():
    obtain_model_data()


if __name__ == "__main__":
    main()
