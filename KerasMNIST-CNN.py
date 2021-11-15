import tensorflow as tf
import numpy as np

( x_train, y_train ), ( x_test, y_test ) = tf.keras.datasets.mnist.load_data()

## The MNIST Data Set is composed of 28x28 black and white pixel images of handwritten digits,
## and the corresponding number that digit represents, 0-9. This is divided into 60,000 training
## instances, and 10,000 testing instances.

train_in = np.reshape( x_train, (-1, 28, 28, 1) ) / 255 ## This ensures that every image is shaped correctly, 28x28 pixel images
test_in = np.reshape( x_test, (-1, 28, 28, 1) ) / 255 ## and is each pixel is scaled to be between 0 and 1.
train_out = tf.keras.utils.to_categorical( y_train, 10 ) ## This takes the digit value, and transforms it into a 1-Hot Encoding,
test_out = tf.keras.utils.to_categorical( y_test, 10 ) ## so that the digit 7 is represented with the vector [0,0,0,0,0,0,0,1,0,0], etc



#digit_input = tf.keras.layers.Input( shape = (28,28,1) )
#cnn_1 = tf.keras.layers.Conv2D( filters = 64, kernel_size = (3,3), strides = (1,1), padding = "valid", activation = tf.nn.relu )( digit_input )
#flatten_image = tf.keras.layers.Flatten()( cnn_1 )
#dropout_1 = tf.keras.layers.Dropout( rate = 0.5 )( flatten_image )
#dense_1 = tf.keras.layers.Dense( units = 50, activation = tf.nn.relu )( dropout_1 )
#logits = tf.keras.layers.Dense( units = 10, activation = None )( dense_1 )
#probabilities = tf.keras.layers.Softmax()( logits )



digit_input = tf.keras.layers.Input( shape = (28,28,1) ) ## Each input image is 28x28 pixels, with 1 channel depth for each (the grayscale value)
cnn_1 = tf.keras.layers.Conv2D( filters = 64, kernel_size = (3,3), strides = (1,1), padding = "valid", activation = tf.nn.relu )( digit_input )
    ## We pass into a 2D convolutional layer, computing 64 spatially invariant features over windows of size 3x3, with a stride of 1 to walk over
    ## every possible 3x3 square in the pixel field, with no padding. The linear combinations of the inputs are passed into a ReLU activation function.
    ## Note, the output of this layer will be a block of size 26x26x64.
flatten_image = tf.keras.layers.Flatten()( cnn_1 )
    ## This flattens the block from the CNN layer to a flat vector of size 43264
dropout_1 = tf.keras.layers.Dropout( rate = 0.5 )( flatten_image )
    ## The 43264 vector is passed through a dropout layer - the output size will still be 43264, but each training step half (rate=0.5)
    ## of the values are `killed' or set to 0.
dense_1 = tf.keras.layers.Dense( units = 50, activation = tf.nn.relu )( dropout_1 )
    ## The 43264 node dropout layer is passed into 50 dense units, with activation of ReLU
logits = tf.keras.layers.Dense( units = 10, activation = None )( dense_1 )
    ## And then into a layer of 10 nodes, one for each class, just taking a linear combination of the inputs as the outputs (no activation)
probabilities = tf.keras.layers.Softmax()( logits )
    ## The unbounded logits are passed into a softmax layer to convert them to 10 exclusive class probabilities.

model = tf.keras.Model( inputs = digit_input, outputs = probabilities )
    ## Builds a model object, taking as input the rows_input object and giving as output the corresponding 10 probabilities for each class
model.compile( optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )
    ## The model is compiled, using the categorical_crossentropy (multi-class log loss error) as the loss function,
    ## and utilizing the adam optimizer to tweak the internal weights/biases to try to minimize loss.

def generate_confusion_matrix( data, labels ):
    ## The confusion matrix represents, for each `true' class, how each data point in that class
    ## was classified. A correct classifier should yield a perfectly diagonal confusion matrix.
    mat = [ [ 0 for i in range(10) ] for j in range(10) ]

    predictions = np.argmax( model.predict( data ), axis = 1 )
    ## model.predict computes 10 probabilities for each data point in data, and then the argmax
    ## determines the index with the largest probability, i.e., the class with the highest likelihood

    for i in range( data.shape[0] ):
        mat[ labels[i] ][ predictions[i] ] += 1
        ## For each true label, increment the predicted value cound

    for i in range(10):
        print( "Actual Digit", i, ":", "\t".join( [ str(c) for c in mat[i] ] ) )

generate_confusion_matrix( test_in, y_test )

input("Enter to Train")

history = model.fit( train_in, train_out, epochs = 3 )
generate_confusion_matrix( test_in, y_test )
