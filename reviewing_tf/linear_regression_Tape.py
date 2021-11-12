# Using Optimizer and Gradient.Tape---------------------------------------------
import numpy as np
import tensorflow as tf

def generate_dataset():
    x_batch = np.linspace(0, 2, 100)
    y_batch = 1.5 * x_batch + 0.5
    return x_batch, y_batch


def linear_model(x):
    return b + tf.multiply(x,w)


def train_step(x, y):
    with tf.GradientTape() as tape:
        predicted = linear_model(x)   
        loss_value = loss_object(y, predicted)
        print(f"Loss Value:{loss_value}")
        grads = tape.gradient(loss_value, [b,w])
        optimizer.apply_gradients(zip(grads, [b,w]))

x, y = generate_dataset()
def train(epochs):
    for epoch in range(epochs):
            train_step(x, y)
    print ('Epoch {} finished'.format(epoch))



optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.MeanSquaredLogarithmicError()
w = tf.Variable(1,dtype=tf.float64)
b = tf.Variable(1,dtype=tf.float64)


train(epochs = 1000)
print(w, b)