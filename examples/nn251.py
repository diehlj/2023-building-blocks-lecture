# code for Example nn251

import tensorflow as tf
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def ex_nn251():
    def create_random_model():
        # Define the model
        model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Dense(32, activation="relu",  kernel_initializer='glorot_normal', input_shape=(2,)))
        model.add(tf.keras.layers.Dense(5, activation="relu",
                                        #  bias_initializer='glorot_normal',
                                         kernel_initializer='glorot_normal',input_shape=(2,)))
        model.add(tf.keras.layers.Dense(1, activation="linear", kernel_initializer='glorot_normal'))

        return model

    def create_plot(model, x_value, row, col):
        y_value = model.predict(x_value).flatten()

        # Create the 3D scatter plot
        trace1 = go.Scatter3d(x=x_value[:, 0], y=x_value[:, 1], z=y_value,
                        mode='markers',
                        marker=dict(size=4, color=y_value, colorscale='Viridis', opacity=0.5, showscale=False),
                        name="Predicted values",
                        showlegend=False)
        
        fig.add_trace(trace1, row=row, col=col)

    # Generate a grid of inputs for visualizing the function
    xx, yy = np.meshgrid(2*np.linspace(-1, 1, 100), 2*np.linspace(-1, 1, 100))
    x_value = np.c_[xx.ravel(), yy.ravel()]

    # Create a subplot with 2 rows and 5 columns
    fig = make_subplots(rows=2, cols=5, specs=[[{'type': 'scatter3d'}]*5]*2)

    # Generate and display 10 different realizations of the network
    for i in range(10):
        row = i // 5 + 1
        col = i % 5 + 1
        print('=================================')
        model = create_random_model()
        print('weights=', model.get_weights())
        create_plot(model, x_value, row, col)

    # Update subplot titles
    for i in range(1, 3):
        for j in range(1, 6):
            fig.update_scenes(row=i, col=j, xaxis_title="X1", yaxis_title="X2", zaxis_title="Product")

    # Update subplot layout
    fig.update_layout(width=2000, height=1000, title="Realizations of Randomly Initialized Neural Networks")
    fig.show()

def ex_nn251_1D():
    def create_random_model():
        # Define the model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(5, activation="relu",
                                        kernel_initializer='glorot_normal',
                                        bias_initializer='glorot_normal',
                                        input_shape=(1,)))
        model.add(tf.keras.layers.Dense(1, activation="linear", kernel_initializer='glorot_normal'))

        return model

    def create_plot(model, x_value, row, col):
        y_value = model.predict(x_value).flatten()

        # Create the 2D scatter plot
        trace1 = go.Scatter(x=x_value[:, 0], y=y_value,
                        mode='markers',
                        marker=dict(size=4, color=y_value, colorscale='Viridis', opacity=0.5, showscale=False),
                        name="Predicted values",
                        showlegend=False)
        
        fig.add_trace(trace1, row=row, col=col)

    # Generate a grid of inputs for visualizing the function
    x_value = np.linspace(-10, 10, 100).reshape(-1, 1)
    # print(x_value.shape)
    # xx

    # Create a subplot with 2 rows and 5 columns
    fig = make_subplots(rows=2, cols=5, specs=[[{'type': 'scatter'}]*5]*2)

    # Generate and display 10 different realizations of the network
    for i in range(10):
        row = i // 5 + 1
        col = i % 5 + 1
        print('=================================')
        model = create_random_model()
        print('weights=', model.get_weights())
        create_plot(model, x_value, row, col)

    # Update subplot titles
    for i in range(1, 3):
        for j in range(1, 6):
            fig.update_xaxes(title_text="X", row=i, col=j)
            fig.update_yaxes(title_text="Output", row=i, col=j)

    # Update subplot layout
    fig.update_layout(width=2000, height=1000, title="Realizations of Randomly Initialized Neural Networks (1D)")
    fig.show()


if __name__ == "__main__":
    ex_nn251()
    ex_nn251_1D()