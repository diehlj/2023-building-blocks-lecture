import tensorflow as tf
from tensorflow.keras import layers
import plotly.express as px
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0 # To have values between 0 and 1; try without it!
x_test  = x_test.astype('float32')  / 255.0

# Plottin some examples:
fig = px.imshow(np.concatenate(x_train[:8], axis=1))
fig.show()

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10) # Look up what this does!
y_test  = tf.keras.utils.to_categorical(y_test, num_classes=10)


def create_fcn_model():
    # TODO Create a model, either
    # using the Sequential API or
    # creating a subclass of tf.keras.Model.
    pass


fcn_model = create_fcn_model()
print(fcn_model.summary())

# Create the losses:
loss_ce = TODO
loss_l1 = TODO
loss_l2 = TODO
loss = loss_ce

optimizer = TODO
fcn_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
history = fcn_model.fit(x_train, y_train, batch_size=128, epochs=2, validation_split=0.2)
test_loss, test_accuracy = fcn_model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")


# TODO Now create a Callback, to log 'loss', 'accuracy' after each batch.
batch_loss_logger = BatchLossLogger()

fcn_model = create_fcn_model()
fcn_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
history = TODO

fig = px.line(x=np.arange(len(batch_loss_logger.accuracies)), y=batch_loss_logger.accuracies)
fig.show()
