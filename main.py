from dataclasses import dataclass
from typing import List
import math, random

@dataclass
class Layer:
    weights: List[float]
    biases: List[float]
    inputs: int
    outputs: int

@dataclass
class Network:
    hidden: Layer
    output: Layer

@dataclass
class InputData:
    images: [bytes]
    labels: [bytes]
    imageCount: int


def load_mnist_images(filename: str):
    with open(filename, 'rb') as f:
        f.read(4) # Read the magic number
        f.read(4) # Read the number of images in the set
        rows = int.from_bytes(f.read(4))
        cols = int.from_bytes(f.read(4))
        
        images = list(iter(lambda: f.read(rows * cols), b''))

        return images

def load_mnist_labels(filename: str):
    with open(filename, 'rb') as f:
        f.read(4) # Read the magic number
        f.read(4) # Read number of labels
        labels = list(iter(lambda: f.read(1), b''))

        return labels

def initialise_layer(layer: Layer, inputs: int, outputs: int):
    n = inputs * outputs
    scale = math.sqrt(2.0 / inputs)

    layer.inputs = inputs
    layer.outputs = outputs

    layer.biases = [0] * outputs

    for i in range(n):
        layer.weights[i] = random.uniform(-1, 1) * scale

def forward_propagation(layer: Layer, inputs: List[float], outputs: List[float]):
    for i in range(layer.outputs):
        outputs[i] = layer.biases[i]
        for j in range(layer.inputs):
            outputs[i] += inputs[j] * layer.weights[j * layer.outputs + i]

def softmax(input: List[float], size: int):
    max = input[0]
    sum = 0

    for i in range(size):
        if (input[i] > max):
            max = input[i]
    
    for i in range(size):
        input[i] = math.exp(input[i] - max)
        sum += input[i]

    for i in range(size):
        input[i] /= sum

def backward_propagation(layer: Layer, input: List[float], output_grad: List[float], input_grad: List[float], lr: float):
    for i in range(layer.outputs):
        for j in range(layer.inputs):
            idx = j * layer.outputs + i
            grad = output_grad[i] * input[j]
            layer.weights[idx] -= lr * grad
            if (input_grad):
                input_grad[j] += output_grad[i] * layer.weights[idx]

        layer.biases[i] -= lr * output_grad[i]

# def train():

inputs = 2
outputs = 1

layer = Layer([0] * inputs * outputs, [0] * outputs, inputs, outputs)

initialise_layer(layer, inputs, outputs)

print(layer)

layer_inputs = [0.5, 0.5]
layer_outputs = [0]

forward_propagation(layer, layer_inputs, layer_outputs)

print(layer_inputs)
print(layer_outputs)



# images = load_mnist_images('train-images.idx3-ubyte')

# labels = load_mnist_labels('train-labels.idx1-ubyte')

