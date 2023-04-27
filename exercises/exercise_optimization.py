import tensorflow as tf

def regression_exercise():
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objs as go

    def f(_x):
      return 3 * np.sin(5*_x) + _x**3 

    x_train = np.random.uniform(-2, 2, 100)
    y_train = f(x_train)

    x_train = tf.expand_dims(x_train, axis=1)
    y_train = tf.expand_dims(y_train, axis=1)

    model = tf.keras.Sequential()

    # Add a (hidden) Dense layer of size 80 with ReLU activation and an (output)
    # dense layer of size 1 without activation:
    TODO

    learning_rate = .001
    # Create an optimizer object with the given learning rate; SGD, Adams, or something else:
    optimizer = TODO
    # Create a loss object; mean square, mean absolute, or something else:
    loss = TODO

    # Compile the model:
    model.compile(optimizer=optimizer, loss=loss)

    epochs     = 10
    batch_size = 16
    # Fit the model.
    TODO

    # Visualize the result:
    x_test = np.linspace(-2, 2, 1000).reshape(-1, 1)
    y_test = model.predict(x_test).flatten()
    y_true = f(x_test).flatten()
    x_test = x_test.flatten()
    trace_true = go.Scatter(x=x_test, y=y_true, mode='lines', name = 'Target function')
    trace_test = go.Scatter(x=x_test, y=y_test, mode='lines', line=dict(color='orange'), name = 'Learned function')
    training_data_trace = go.Scatter(x=x_train.numpy().flatten(), y=y_train.numpy().flatten(), mode='markers', name='Training Data', marker=dict(color='magenta', size=8), opacity=0.3)
    layout = go.Layout(xaxis=dict(title='x'), yaxis=dict(title='f(x)'), title="Function: 3 * sin(5*x) + x^3 vs FFN")
    fig = go.Figure(data=[trace_true,trace_test,training_data_trace], layout=layout)
    fig.show()

    # TODO
    # Make the learning_rate, nr of training samples, the optimizer class, nr of epochs and batch size,
    # all parameters of this function and experiment with changing these hyperparameters.
    # Can you find a combination that works well?
    # Can you find a combination that produces NaNs?

def gradient_descent_exercise():
    import numpy as np
    import plotly.subplots as sp
    import plotly.graph_objs as go

    def f(x):
        return x ** 2

    def gradient_f(x):
        return 2 * x

    learning_rate  = 0.9
    x_start        = 5.1
    num_iterations = 15

    x_values = [x_start]
    y_values = [f(x_start)]

    x_current = x_start
    for i in range(num_iterations):
        # Do a gradient descent step:
        x_current = TODO
        x_values.append(x_current)
        y_values.append(f(x_current))

    # Plotting the results (you don't need to understand this code but might have to change X_MIN and X_MAX):
    X_MIN = np.min(x_values) - 5.
    X_MAX = np.max(x_values) + 5.
    x = np.linspace(X_MIN, X_MAX, 100)
    y = f(x)
    fig = sp.make_subplots(rows=1, cols=1)
    function_trace = go.Scatter(x=x, y=y, mode='lines', name='f(x) = x^2')
    fig.add_trace(function_trace)
    fig.add_trace(go.Scatter(x=[x_values[0]], y=[y_values[0]], mode='markers', marker=dict(color='red'), name='Gradient Descent'), row=1, col=1)
    frames = [go.Frame(data=[function_trace, go.Scatter(x=x_values[:i + 1], y=y_values[:i + 1], mode='markers', marker=dict(color='red'))]) for i in range(1, len(x_values))]
    fig.frames = frames
    animation_settings = dict(frame=dict(duration=500, redraw=True), fromcurrent=True)
    fig.update_layout(updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play', method='animate', args=[None, animation_settings])])],
                    xaxis_title='x', yaxis_title='f(x)', title='Gradient Descent Visualization',
                    xaxis=dict(range=[X_MIN, X_MAX]),
                    yaxis=dict(range=[np.min(y), np.max(y)]))
    fig.show()



def layer_normalization_exercise():
    # Step 1: Set the random seed for TensorFlow.
    TODO

    # Step 2: Generate random input data of shape (2,3).
    data = TODO

    # Step 3: Initialize a LayerNormalization object.
    ln = tf.keras.layers.LayerNormalization()

    # Step 4: Calculate the mean and variance of the input data.
    m  = TODO
    v  = TODO

    # Step 5: Perform layer normalization on the input data.
    norm = ln(data, training=True)
    my_norm = TODO

    # Step 6: Use TensorFlow's debugging tools to assert that the layer normalization is performed correctly.
    tf.debugging.assert_near(norm, my_norm, atol=1e-3)

    # Step 7: do the same for input data of shape (2,3,4).
    TODO

def parametrization_matters_exercise():
    # random matrix:
    n = 10
    A = tf.random.normal( (n,n) )
    def f(_x):
        return tf.einsum('ij,bj->bi', A, _x)
    n_samples = 10_000
    x_train = tf.random.normal( (n_samples, n) )
    y_train = f(x_train)
    # Create three models:
    # - a linear, one-layer model Dense_{A,b;\id}; A \in \R^{n \times n}, b \in \R^n
    # - a linear, two-layer model Dense_{A_2,b_2;\id} \circ Dense_{A_1,b_1;\id}; A_i \in \R^{n \times n}, b_i \in \R^n
    # - a linear, two-layer model Dense_{A_2,b_2;\id} \circ Dense_{A_1,b_1;\id};
    #   A_1 \in \R^{m \times n}, b_1 \in \R^m, A_2 \in \R^{n \times m}, b_2 \in \R^n; for some m < n
    # 
    # Train each model and compare the results (convergence, generalization, etc.).

if __name__ == '__main__':
    # gradient_descent_exercise()
    # regression_exercise()
    parametrization_matters_exercise()
