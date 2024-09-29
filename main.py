from dataclasses import dataclass
from typing import List
import math
import random

# Neural Network Configuration
INPUT_SIZE = 784
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10

LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 60

class Layer:
    weights: List[float]
    biases: List[float]
    inputs: int
    outputs: int

    def __init__(self, inputs: int, outputs: int):
        # Nw = Ni x No
        n = inputs * outputs

        # Calculate what standard deviation should be, based on no. of inputs
        scale = math.sqrt(2.0 / inputs)

        self.inputs = inputs
        self.outputs = outputs

        self.biases = [0] * outputs
        self.weights = [0] * n

        # Set the default value for weights using Kaiming Initialisation
        # Take a random number from a normal distribution and apply the SD from before
        for i in range(n):
            self.weights[i] = random.uniform(-1, 1) * scale

@dataclass
class Network:
    hidden: Layer
    output: Layer

@dataclass
class InputData:
    images: [[float]]
    labels: [int]
    imageCount: int


def load_mnist_images(filename: str) -> (int, [[float]]):
    with open(filename, 'rb') as f:
        # Read the magic number
        f.read(4)

        # Read the number of images in the set, and resolution of images
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        
        # Read all images into list of bytes
        images = list(iter(lambda: f.read(rows * cols), b''))

        # Convert the image (a list of 8-bit integers) into a list of floats
        float_images = [[(byte / 255.00) for byte in image] for image in images]
                
        return (num_images, float_images)


def load_mnist_labels(filename: str) -> (int, [int]):
    with open(filename, 'rb') as f:
        # Read the magic number
        f.read(4)

        # Read number of labels, read all labels into list of bytes
        num_labels = int.from_bytes(f.read(4), "big") 
        labels = list(iter(lambda: f.read(1), b''))

        labels = [int.from_bytes(label, "big") for label in labels]

        return (num_labels, labels)

def forward_propagation(layer: Layer, inputs: List[float]) -> List[float]:
    # Create an outputs list of length layer.outputs
    outputs = [0] * layer.outputs
    
    # Loop over each output neuron
    for i in range(layer.outputs):
        # Start with the bias value for the output
        outputs[i] = layer.biases[i]
        # Loop over input (j) into the output neuron (i)
        for j in range(layer.inputs):
            # For each input J into the neuron, add on its weighted value
            # The weight for the j'th input to the i'th output neuron is taken from the weights matrix
            outputs[i] += inputs[j] * layer.weights[j * layer.outputs + i]

    # Return the weighted outputs
    return outputs


def softmax(input: List[float]) -> List[float]:
    # Find the biggest input
    max_input = max(input)
    # Initialise sum and output list
    sum = 0
    output = [0.00] * len(input)
    
    # Loop over all inputs
    for idx, value in enumerate(input):
        # Calculate e^(x - max), this ensures each output value is between 0 and 1
        output[idx] = math.exp(value - max_input)
        sum += output[idx]

    # Loop over all outputs, dividing them by the sum value
    # This makes the sum of all output values = 1
    for i in range(len(output)):
        output[i] /= sum
    
    return output


def backward_propagation(layer: Layer, input: List[float], output_grad: List[float], learning_rate: float) -> List[float]:
    # Initialise the input gradient list
    input_grad = [0.00] * len(input)
    
    # For all outputs in the layer
    for i in range(layer.outputs):
        # Loop over all inputs to the layer
        for j in range(layer.inputs):
            # Calculate the index of the weight for the connection between output i and input j
            idx = j * layer.outputs + i
            # Calculate the gradient for that specific connection
            grad = output_grad[i] * input[j]
            # Update the weight for that connection, applying a learning rate to ensure the network doesn't change a weight too much at once
            # Changing the weights too much at once can cause training to get worse
            layer.weights[idx] -= learning_rate * grad

            # Update the input gradients to reflect the loss with respect to inputs of the layer
            input_grad[j] += output_grad[i] * layer.weights[idx]

        # Update the output biases, accounting for learning rate
        layer.biases[i] -= learning_rate * output_grad[i]

    return input_grad


def train(net: Network, input: [float], label: int, learning_rate: float):
    final_output = [0] * OUTPUT_SIZE
    hidden_output = [0] * HIDDEN_SIZE

    output_grad = [0] * OUTPUT_SIZE
    hidden_grad = [0] * HIDDEN_SIZE

    # Forward Pass: Input to the Hidden layer
    hidden_output = forward_propagation(net.hidden, input)

    # ReLU activation function
    hidden_output = [(output if output > 0 else 0) for output in hidden_output]

    # Forward Pass: Hidden to the Output layer
    final_outout = forward_propagation(net.output, hidden_output)
    final_outout = softmax(final_output)

    # Compute the Output Gradient
    for i in range(OUTPUT_SIZE):
        output_grad[i] = final_output[i] - (i == label)
    
    # Backward propagation pass: Output Layer to Hidden Layer
    hidden_grad = backward_propagation(net.output, hidden_output, output_grad, learning_rate)

    # Backpropagate through the ReLU activation function
    for i in range(HIDDEN_SIZE):
        hidden_grad[i] *= hidden_output[i] if hidden_output[i] > 0 else 0

    backward_propagation(net.hidden, input, hidden_grad, learning_rate)


def predict(net: Network, input: [float]) -> int:
    hidden_output = [0] * HIDDEN_SIZE
    final_output = [0] * OUTPUT_SIZE

    forward_propagation(net.hidden, input, hidden_output)
    for i in range(HIDDEN_SIZE):
        hidden_output[i] = hidden_output[i] if hidden_output[i] > 0 else 0

    forward_propagation(net.output, hidden_output, final_output)
    softmax(final_output, OUTPUT_SIZE)

    # Valid predictions are 0-9, same as the indexes of the final_output list
    # Get the index of the output with the highest weight, that's the predicted answer
    max_index = final_output.index(max(final_output))

    return max_index

def main():
    num_images, training_images = load_mnist_images("train-images.idx3-ubyte")
    _, training_labels = load_mnist_labels("train-labels.idx1-ubyte")

    training_data = InputData(training_images, training_labels, num_images)

    num_test_images, test_images = load_mnist_images("t10k-images.idx3-ubyte")
    _, test_labels = load_mnist_labels("t10k-labels.idx1-ubyte")

    test_data = InputData(test_images, test_labels, num_test_images)

    hidden_layer = Layer(INPUT_SIZE, HIDDEN_SIZE)
    output_layer = Layer(HIDDEN_SIZE, OUTPUT_SIZE)

    net = Network(hidden_layer, output_layer)

    for epoch in range(EPOCHS):
        total_loss = 0.00
        for i in range(0, len(training_images), BATCH_SIZE):
            for j in range(0, BATCH_SIZE):
                idx = i + j
                
                train(net, training_data.images[idx], training_data.labels[idx], LEARNING_RATE)

                # hidden_output = [0.00] * HIDDEN_SIZE
                # final_output = [0.00] * OUTPUT_SIZE

                hidden_output = forward_propagation(net.hidden, training_data.images[idx])

                for k in range(HIDDEN_SIZE):
                    hidden_output[k] = hidden_output[k] if hidden_output[k] > 0 else 0
                
                final_output = forward_propagation(net.output, hidden_output)
                final_output = softmax(final_output)

                total_loss += -math.log(final_output[training_data.labels[idx]] + 1e-10)
            
        correct = 0
        for i in range(num_test_images):
            if (predict(net, test_data.images[i]) == test_data.labels[i]):
                correct += 1
        
        print(f"Epoch {epoch + 1}, Accuracy: {correct / num_test_images * 100.00}, Avg Loss: {total_loss / num_images}")

if __name__ == '__main__':
    main()

# inputs = 2
# outputs = 1

# layer = Layer([0] * inputs * outputs, [0] * outputs, inputs, outputs)

# initialise_layer(layer, inputs, outputs)

# print(layer)

# layer_inputs = [0.5, 0.5]
# layer_outputs = [0]

# forward_propagation(layer, layer_inputs, layer_outputs)

# print(layer_inputs)
# print(layer_outputs)

