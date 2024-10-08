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

    :param layer: The Layer to be acted upon
    :param inputs: The inputs to the layer

    :type layer: Layer
    :type inputs: np.ndarray

    :returns A 1-D array containing the pre-activation outputs of the Layer
    :rtype: np.ndarray
    """

    # Forward propagation computes the output of a layer given its inputs.
    # In a fully connected layer, each output is the weighted sum of *all* inputs
    # The weights represent how much each input contributes to each neuron.
    # This can be done for all output neurons at once using matrix multiplication.
    # This results in an output array of neurons which the biases can then be added on to,
    # to increase the weight of each output prior to activation
    #
    # The transpose is taken as the weights matrix is in the format (inputs, outputs)
    # and the inputs array is in the format (inputs, 1) - for matrix multiplication,
    # the format must be (outputs, inputs) to match the size of (inputs, 1)
    return np.dot(inputs, layer.weights.transpose()) + layer.biases

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def backward_propagation(layer: Layer, input: np.ndarray, output_grad: np.ndarray, learning_rate: float) -> np.ndarray:
    input_grad = np.dot(output_grad, layer.weights)
    layer.weights -= learning_rate * np.dot(output_grad.T, input)
    layer.biases -= learning_rate * np.sum(output_grad, axis=0)
    return input_grad

def train(net: Network, input: np.ndarray, label: np.ndarray, learning_rate: float):
    # Forward pass
    hidden_output = relu(forward_propagation(net.hidden, input))
    final_output = softmax(forward_propagation(net.output, hidden_output))

    # Compute gradients
    output_grad = final_output - label
    hidden_grad = backward_propagation(net.output, hidden_output, output_grad, learning_rate)
    hidden_grad *= (hidden_output > 0)

    backward_propagation(net.hidden, input, hidden_grad, learning_rate)

def predict(net: Network, input: np.ndarray) -> np.ndarray:
    hidden_output = relu(forward_propagation(net.hidden, input))
    final_output = softmax(forward_propagation(net.output, hidden_output))
    return np.argmax(final_output, axis=1)

def main():
    num_images, training_images = load_mnist_images("train-images.idx3-ubyte")
    _, training_labels = load_mnist_labels("train-labels.idx1-ubyte")
    training_data = InputData(training_images, training_labels, num_images)

    num_test_images, test_images = load_mnist_images("t10k-images.idx3-ubyte")
    _, test_labels = load_mnist_labels("t10k-labels.idx1-ubyte")
    test_data = InputData(test_images, test_labels, num_test_images)

    net = Network(Layer.initialize(INPUT_SIZE, HIDDEN_SIZE), Layer.initialize(HIDDEN_SIZE, OUTPUT_SIZE))

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for i in range(0, num_images, BATCH_SIZE):
            batch_images = training_data.images[i:i+BATCH_SIZE]
            batch_labels = training_data.labels[i:i+BATCH_SIZE]
            
            one_hot_labels = np.eye(OUTPUT_SIZE)[batch_labels]
            train(net, batch_images, one_hot_labels, LEARNING_RATE)

            hidden_output = relu(forward_propagation(net.hidden, batch_images))
            final_output = softmax(forward_propagation(net.output, hidden_output))
            
            # Use np.clip to avoid log(0)
            log_probs = np.log(np.clip(final_output[np.arange(len(batch_labels)), batch_labels], 1e-10, 1.0))
            total_loss -= np.sum(log_probs)

        predictions = predict(net, test_data.images)
        accuracy = np.mean(predictions == test_data.labels)
        
        print(f"Epoch {epoch + 1}, Accuracy: {accuracy * 100:.2f}%, Avg Loss: {total_loss / num_images:.4f}")

if __name__ == '__main__':
    main()