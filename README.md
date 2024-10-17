# miniMNIST-python
This project is an implementation of the neural network found in [miniMNIST-c](https://github.com/konrad-gajdus/miniMNIST-c), in Python, with NumPy.
It is a minimal neural network for classifying handwritten digits from the MNIST dataset, and the entire implementation is **87 lines of code** according to `cloc`.
Unlike [miniMNIST-c](https://github.com/konrad-gajdus/miniMNIST-c), this project makes use of one library: [NumPy](https://numpy.org/).
NumPy is used for its powerful N-dimensional arrays which are extremely fast and allow the entire network to be vectorised and trained rapidly.
It also makes translating the mathematics behind a simple feed-foward neural network into Python much easier as it provides useful functions such as the matrix dot product, the normal distribution, and the argmax function.

## Features
* Two-layer Neural Network (Input -> Hidden -> Output)
* ReLU activation for the Hidden Layer and SoftMax activation for the Output Layer
* Cross-entropy Loss function (Log Loss)
* Stochastic Gradient Descent (SGD) optimizer

## Differences
* The code for `miniMNIST-python` is commented to an almost extreme degree, as this implementation was developed for a workshop presented by [CompSoc - University of Galway's Computer Society](https://github.com/ugcompsoc) on 2024/10/16
* Many of the optimisations made to `miniMNIST-c` since its initially release have not been implemented in `miniMNIST-python`
* Does not implement Momentum-based variation of Stochastic Gradient Descent
* Utilises the `t10k` testing dataset rather than taking a slice of the `MNIST` training dataset for testing

## Prerequisites and Usage
* Python 3.12
* NumPy
* MNIST dataset files:
  - train-images.idx3-ubyte
  - train-labels.idx1-ubyte
  - t10k-images.idx3-ubyte
  - t10k-labels.idx1-ubyte

1. Place the MNIST dataset files in the same directory as `main.py` (the root of the project)
2. Install NumPy:
```bash
pip install numpy
```
3. Execute the program with Python 3.12:
```bash
python main.py
```
The script will train the neural network and output the accuracy and average loss for each training epoch.

## Configuration Options
The constants at the top of `main.py` can be adjusted to change the behaviour of the network, namely:
- `HIDDEN_SIZE`: The number of neurons in the Hidden Layer
- `LEARNING_RATE`: The learning rate for Stochastic Gradient Descent
- `EPOCHS`: The number of training epochs
- `BATCH_SIZE`: The batch size for training (in this implementation it must be a number which divides cleanly into 60,000)

## License
This project is open-source and available under the [MIT License](https://github.com/daxorinator/miniMNIST-python/blob/main/LICENSE).