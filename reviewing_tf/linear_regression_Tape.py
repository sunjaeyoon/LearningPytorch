# Using Optimizer and Gradient.Tape---------------------------------------------
import numpy as np
import tensorflow as tf

w = tf.Variable(1,dtype=tf.float64)
b = tf.Variable(1,dtype=tf.float64)

def linear_model(x):
    return b + tf.multiply(x,w)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.MeanSquaredLogarithmicError()

def train_step(x, y):
    with tf.GradientTape() as tape:
        predicted = linear_model(x)   
        loss_value = loss_object(y, predicted)
        print(f"Loss Value:{loss_value}")
        grads = tape.gradient(loss_value, [b,w])
        optimizer.apply_gradients(zip(grads, [b,w]))

def train(epochs):
    for epoch in range(epochs):
            train_step(x, y)
    print ('Epoch {} finished'.format(epoch))

train(epochs = 1000)

print(w, b)
