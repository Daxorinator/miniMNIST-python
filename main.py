import numpy as np
from dataclasses import dataclass

# Neural Network Configuration
INPUT_SIZE = 784
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10

LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 60

@dataclass
class Layer:
    """
    Represents a layer in a feed-forward neural network

    :ivar weights: The weight of each connection between a layer input and a layer output
    :type weights: np.ndarray

    :ivar biases: The bias added to each output of the layer
    :type biases: np.ndarray
    """
    weights: np.ndarray
    biases: np.ndarray

    @classmethod
    def initialize(cls, inputs: int, outputs: int):
        """
        Returns a new Layer with the specified number of inputs and outputs.
        The biases are initialised to zero, and the weights are initialised using the Kaiming Initialisation method

        :param inputs: The number of inputs to the layer
        :param outputs: The number of outputs from the layer
        :returns: A new Layer with weights and biases
        :rtype: Layer
        """

        # Kaiming Initialisation:
        # Initialise the weights by sampling a normal distribution with a
        # variation of 2/Ni, or in standard deviation form, root(2/Ni)
        stddev = np.sqrt(2.0 / inputs)
        weights = np.random.normal(loc=0.0, scale=stddev, size=(inputs, outputs))

        # Biases should be initialised to zero so the network can adjust them during training
        biases = np.zeros(outputs)
        return cls(weights, biases)

@dataclass
class Network:
    """
    A Neural Network with one hidden layer and one output layer
    :ivar hidden: The Hidden Layer
    :ivar output: The Output Layer
    :type hidden: Layer
    :type output: Layer
    """
    hidden: Layer
    output: Layer

@dataclass
class InputData:
    """
    Input data for the Neural Network, consisting of images, labels, and the number of images in the dataset
    :ivar images: An array of images
    :ivar labels: An array of labels
    :ivar imageCount: The number of images in `images`
    :type images: np.ndarray
    :type labels: np.ndarray
    :type imageCount: int
    """
    images: np.ndarray
    labels: np.ndarray
    imageCount: int

def load_mnist_images(filename: str) -> (int, np.ndarray):
    """
    Loads an MNIST image dataset (can be test or train)
    Transforms each pixel in each image into a float between 0.0 and 1.0
    :param filename: File or File Path containing MNIST images
    :type filename: str

    :returns A tuple containing the number of images as an int and the image array as an np.ndarray
    :rtype: (int, np.ndarray)
    """
    with open(filename, 'rb') as f:
        # Interpret the 16 bytes from f.read(16) as 4 big-endian integers
        # The first one (ignored) is the magic number, the next 3 are 
        # the number of images, and number of vertical and horizontal pixels
        _, num_images, rows, cols = np.frombuffer(f.read(16), dtype='>i4')

        # Interpret every byte after that as a grayscale pixel, represented by an unsigned 8-bit integer
        # i.e. a value between 0 and 255 - then reshape the array of integers into a 2 dimensional array
        # of size num_images x rows*cols, i.e. a bunch of arrays with rows*cols pixels in them
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)

        # Neural Nets usually work better when their inputs are floats, all of the integer pixel values
        # can be converted into floats between 0 and 1 by dividing them by 255

        # Return the number of images, and the 2D array of images with pixel values expressed as floats
        return num_images, images.astype(np.float32) / 255.0

def load_mnist_labels(filename: str) -> (int, np.ndarray):
    """
    Loads an MNIST label dataset (can be test or train)
    All labels are loaded as 8-bit unsigned integers

    :param filename: File or Fiel Path containing MNIST labels
    
    :returns A tuple containing the number of labels as an int and the label array as an np.ndarray
    :rtype: (int, np.ndarray)
    """
    with open(filename, 'rb') as f:
        # Read (and ignore) the magic number, and number of labels
        _, num_labels = np.frombuffer(f.read(8), dtype='>i4')
        
        # Read the rest of the bytes as unsigned 8-bit integers,
        # each label is only 1 byte as labels are 0-9
        labels = np.frombuffer(f.read(), dtype=np.uint8)

        return num_labels, labels

def forward_propagation(layer: Layer, inputs: np.ndarray) -> np.ndarray:
    """
    Performs forward propagation on the specified Layer with the provided inputs and returns the pre-activation outputs of the layer.

    Forward propagation computes the output of a layer given its inputs.
    In a fully connected layer, each output is the weighted sum of *all* inputs
    The weights represent how much each input contributes to each neuron.
    This can be done for all output neurons at once using matrix multiplication.
    This results in an output array of neurons which the biases can then be added on to,
    to increase the weight of each output prior to activation
    
    The transpose is taken as the weights matrix is in the format (inputs, outputs)
    and the inputs array is in the format (inputs, 1) - for matrix multiplication,
    the format must be (outputs, inputs) to match the size of (inputs, 1)

    :param layer: The Layer to be acted upon
    :param inputs: The inputs to the layer

    :type layer: Layer
    :type inputs: np.ndarray

    :returns A 1-D array containing the pre-activation outputs of the Layer
    :rtype: np.ndarray
    """

    return np.dot(inputs, layer.weights) + layer.biases

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Applies the softmax activation function to the input array, and normalises the output.

    The e^x function can generate incredibly high numbers if the input values are high - sometimes resulting in Infinity or an overflow.
    This can be avoided by subtracting the maximum value of the input array, from all values in the array.
    This works because x^0 = 1, and e^-y approaches 1 for extremely small values of y, and 0 for extremely large values of y.

    This results in the "feature scaling" of the values so they exist only between 0 and 1.

    The entries in the resulting array are then divided by its sum to produce a valid probability distribution,
    i.e. they are normalised, so that the resulting class with the highest probability can be picked from the array.

    :param x: The pre-activation output array of a neural network layer
    :type x: np.ndarray

    :returns: A probability distribution containing the normalised probabilities of the input values
    :rtype: np.ndarray
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x: np.ndarray) -> np.ndarray:
    """
    Applies the Rectified Linear Unit (ReLU) activation function to the input array.
    If the entries in the input array are greater than 0 they are returned, untouched.
    If the values are less than 0, then they are clamped to 0.
    """
    return np.maximum(0, x)

def backward_propagation(layer: Layer, input: np.ndarray, output_grad: np.ndarray, learning_rate: float) -> np.ndarray:
    """
    Performs backward propagation on the specified Layer with the provided inputs and output gradient of the layer, and returns the input gradient of the layer.

    Backward propagation updates the weights and biases of the specified layer of the network.
    It does this based on the derivative (gradient) of the "loss function" (a measure of 'how wrong' the neural network is),
    with respect to the outputs. The gradient is used because it provides the "direction" where loss increases the most.
    Adjusting in the opposite correction will provide the "most improvement for your input" so to speak.

    This technique is called Gradient Descent minimisation and can be used to find the minimum of the loss function.
    In practise, achieving the minimum would cause overfitting of the neural network, so Gradient Descent is applied
    through training, until the loss value is sufficiently low.

    The weights are updated based on the loss with respect to the weights, rather than the loss with respect to the outputs.
    The loss with respect to the weights can be calculated by multiplying the loss with respect to the outputs, by the inputs.
    The resulting matrix multiplication represents the gradient with respect to the weights - the gradients can then be subtracted
    from the weights to adjust them. 

    The layer biases are updated based on the gradient of the loss, with respect to the biases.
    The biases are applied independent of the outputs, so their gradients are simply the sum of the output gradients

    Lastly, the gradient of loss with respect to the inputs is calculated by taking the dot product of the output gradient,
    and the layer weights - similar to how loss gradient with respect to weights is calculated. Calculating this is useful
    because the inputs to the final layer, are the outputs from the hidden layer - so the input gradient can be used to
    further back propagate in a multi-layer neural network.

    :param layer: The layer to back propagate
    :type layer: Layer

    :param input: The layer inputs which led to the output gradient
    :type input: np.ndarray

    :param output_grad: The output gradient for the layer given the inputs specified in `input`
    :type output_grad: np.ndarray

    :param learning_rate: The learning rate of the neural network - prevents the network weights from wildly changing
    :type learning_rate: float

    :returns: The loss gradient with respect to the layer inputs, based on the provided loss gradient with respect to the layer outputs
    :rtype: np.ndarray
    """
    layer.weights -= np.dot(input.transpose(), output_grad) * learning_rate

    layer.biases -= np.sum(output_grad, axis=0) * learning_rate

    input_grad = np.dot(output_grad, layer.weights.transpose())

    return input_grad

def train(net: Network, input: np.ndarray, label: np.ndarray, learning_rate: float):
    """
    Trains the provided neural network with the given input array and label array, at the specified learning rate.

    :param net: The network to be trained
    :type net: Network

    :param input: The array of inputs to use for training
    :type input: np.ndarray

    :param label: The array of true labels to use for training
    :type label: np.ndarray

    :param learning_rate: The learning rate of the network which slows learning to a reasonable rate
    :type learning_rate: float
    """

    # Forward pass where the hidden layer receives the images and is activated by the ReLU activation function
    hidden_output = relu(forward_propagation(net.hidden, input))
    # Foward pass where the output layer receives the hidden layer output and is activated by the SoftMax activation function
    final_output = softmax(forward_propagation(net.output, hidden_output))

    # Compute the output loss gradient using the Cross Entropy Loss function
    # The gradient of the cross entropy loss with respect to the output *coincidentally* works out
    # as the final outputs of the network, minus the true labels in their one-hot form (see: one-hot encoding / identity matrix) 
    output_grad = final_output - label
    
    # The gradient of the hidden layer output can be gotten using back propagation with the output layer gradient
    hidden_grad = backward_propagation(net.output, hidden_output, output_grad, learning_rate)

    # Only adjust the neurons in the hidden layer which were actually included in the output, i.e. those which were greater than zero (see: ReLU activation)
    # The expression in the brackets returns a boolean (True or False, 1 or 0), the neurons which are not >0 will be multipled by zero and thus won't be modified    
    hidden_grad *= (hidden_output > 0)

    # Perform backward propagation on the hidden layer using the hidden layer output gradient
    backward_propagation(net.hidden, input, hidden_grad, learning_rate)

def predict(net: Network, input: np.ndarray) -> np.ndarray:
    """

    :param net: The network to generate a prediction with
    :type net: Network

    :param input: The input to the neural network
    :type input: np.ndarray

    :returns: An array containing the predictions for the provided inputs (or a single prediction if only one input was given)
    :rtype: np.ndarray
    """

    # Use forward propagation to get the pre-activation output of the Hidden layer from the Input
    # and apply the ReLU activation function to the output
    hidden_output = relu(forward_propagation(net.hidden, input))

    # Use forward propagation to get the pre-activation output of the Output player from the Hidden layer output
    # and apply the SoftMax activation function to the output
    final_output = softmax(forward_propagation(net.output, hidden_output))

    # Return the index of the output neuron with the highest probability
    # For this net, the returned index and the actual category are the same thing (because the prediction is a number from 0-9)
    # For a net that predicts a non-numeric category, e.g. a type of flower, this index can be used to access a one-hot encoded list of labels
    return np.argmax(final_output, axis=1)

def main():
    # Load the training images and the number of training images in the dataset
    num_images, training_images = load_mnist_images("train-images.idx3-ubyte")
    # Load the training labels
    _, training_labels = load_mnist_labels("train-labels.idx1-ubyte")
    
    # Instantiate an InputData struct containing the training data
    training_data = InputData(training_images, training_labels, num_images)

    # Repeat above for the test data
    num_test_images, test_images = load_mnist_images("t10k-images.idx3-ubyte")
    _, test_labels = load_mnist_labels("t10k-labels.idx1-ubyte")
    test_data = InputData(test_images, test_labels, num_test_images)

    # Instantiate a new Neural Network, with two new layers based on the parameters at the top of the file
    net = Network(
        hidden=Layer.initialize(inputs=INPUT_SIZE, outputs=HIDDEN_SIZE),
        output=Layer.initialize(inputs=HIDDEN_SIZE, outputs=OUTPUT_SIZE)
    )

    # Loop for the number of training epochs (an epoch is a training session that runs until the entire training dataset is consumed)
    for epoch in range(EPOCHS):
        # Instantitate the loss figure
        total_loss = 0.0
        # Perform batch training using the batch size set at the top of the file
        for i in range(0, num_images, BATCH_SIZE):
            # Take a sample of the training data and training labels based on the number of images trained, and the size of the batch
            batch_images = training_data.images[i:i+BATCH_SIZE]
            batch_labels = training_data.labels[i:i+BATCH_SIZE]
            
            # Create a one-hot encoded array of true labels present in the batch
            # A one-hot encoded array is the same as an I-matrix
            # This effectively creates a matrix that mimics the output of the neural network, so its easier to calculate loss and check if predictions are correct
            one_hot_labels = np.eye(OUTPUT_SIZE)[batch_labels]

            # Train the network with the images from the batch at the specified learning rate
            train(net, batch_images, one_hot_labels, LEARNING_RATE)

            # Perform a forward propagation pass with the images in the batch,
            # producing an output that reflects how "bad" the network is at the end of the batch
            # This output is used to calculate Loss, an actual reflection of how good or bad the network is
            hidden_output = relu(forward_propagation(net.hidden, batch_images))
            final_output = softmax(forward_propagation(net.output, hidden_output))
            
            # This line is a bit complicated because it contains a few tricks to prevent math errors
            #
            # Firstly range(len(batch_labels)) returns a list in the format [0, 1, 2, ..., n-1] i.e. an array where each entry is an index of batch_labels
            # This is used to index final_output which is a 2D array by passing 2 indices separated by a comma:
            #   First, a range is passed in, this creates a slice of the original final_output containing only the indices relevant to this batch
            #   Secondly, each element in the slice (keep in mind, those elements would be arrays), is indexed using batch_labels which contains the true label for each input
            #
            # This returns a list the length of BATCH_SIZE, where each entry contains the probability of the true label for the given image in batch_images. 
            # This is then clipped so that every entry has a minimum value of 1x10^-10, and maximum value of 1.0 - values outside will break log(x) or be invalid in the case of values >1
            # The log is then taken to calculate the true loss (this comes from the log term in the Cross-Entropy Loss function)
            log_probs = np.log(np.clip(final_output[range(len(batch_labels)), batch_labels], 1e-10, 1.0))
            # This minimics the summation in the definition of Cross-Entropy Loss and adds the loss of this batch onto the loss from other batches
            total_loss -= np.sum(log_probs)

        # Run predictions on the test data
        predictions = predict(net, test_data.images)
        # Take the result and check how many of the results were right - this looks like [True True False False True False True] etc.
        # Or you could represent it as integers: [1 1 0 0 1 0 1]
        # and then take the mean (average) of that, to show the average correctness (accuracy) of the network 
        accuracy = np.mean(predictions == test_data.labels)
        
        # Finally (at last!) print the current epoch number, and the accuracy and loss values that were calculated
        print(f"Epoch {epoch + 1}, Accuracy: {accuracy * 100:.2f}%, Avg Loss: {total_loss / num_images:.4f}")

if __name__ == '__main__':
    main()