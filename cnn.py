from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical


def create_model() -> Sequential:
    """
        Create our NN with two conv layers and two dense layers.

        Returns:
            A compiled keras sequential model ready to be fit.
    """
    # Size of our image
    input_shape = (28, 28, 1)

    # Create our model and add layers
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    # Compile our model
    model.compile(
        loss=categorical_crossentropy, optimizer=SGD(lr=0.01), metrics=["accuracy"]
    )

    return model


def obtain_model_data() -> tuple:
    """
        Obtain the mnist dataset and apply any necessary
        transformations and scaling where needed.

        Returns:
            A tuple of tuples containing our training and testing data
            in the format mnist.load_data() would return it to us.
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape our training and testing data into a 4d tensor
    # (samples, image x, image y, and channels (colors))
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # Convert X train and test data into floats for division
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    # Divide all values by 255 to obtain a decimal value
    X_train /= 255
    X_test /= 255

    # Create our categorical matricies (10 classes)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Return our data
    return (X_train, X_test), (y_train, y_test)


def main():
    """
         Execute the nn and then evaluate the model
    """
    (X_train, X_test), (y_train, y_test) = obtain_model_data()
    model = create_model()

    batch_size = 128
    epochs = 10

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    model.save("mnist.h5")

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == "__main__":
    main()
