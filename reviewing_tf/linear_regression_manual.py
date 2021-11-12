# Manual Linear Regression +optimization in TF 

import numpy as np
import tensorflow as tf

def generate_dataset():
    x_batch = np.linspace(0, 2, 100)
    y_batch = 1.5 * x_batch + 0.5
    return x_batch, y_batch

def mean_squared_error( Y , y_pred ):
    return tf.reduce_mean( tf.square( y_pred - Y ) )

def mean_squared_error_deriv( Y , y_pred ):
    return tf.reshape( tf.reduce_mean( 2 * ( y_pred - Y ) ) , [ 1 , 1 ] )
    
def h ( X , weights , bias ):
    return tf.multiply( X , weights) + bias 

x, y = generate_dataset()

num_features = 1
weight = tf.random.normal( ( num_features , 1 ) )
bias = tf.random.normal( ( num_features , 1 ) )


epochs_plot = list()
loss_plot = list()

num_epochs = 100
learning_rate = 0.01

for i in range( num_epochs ) :
    
    epoch_loss = list()
    for b in range( x.shape[0] ):
   
        output = h( x[b] , weight , bias ) 

        # Calculate loss and gradients
        loss = epoch_loss.append( mean_squared_error( y[b] , output ).numpy() )
        dJ_dH = mean_squared_error_deriv( y[b] , output)
        dH_dW = x[b]
        dJ_dW = tf.reduce_mean( dJ_dH * dH_dW )
        dJ_dB = tf.reduce_mean( dJ_dH )
    
        # Manual Optimization step
        weight -= ( learning_rate * dJ_dW )
        bias -= ( learning_rate * dJ_dB ) 
        
    loss = np.array( epoch_loss ).mean()
    epochs_plot.append( i + 1 )
    loss_plot.append( loss ) 
    
    print( 'Loss is {}'.format( loss ) ) 


print(weight, bias)

