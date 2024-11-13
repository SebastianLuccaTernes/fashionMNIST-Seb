# fashionMNIST-Seb
## Project Overview
This project uses the Fashion MNIST dataset to build and train a neural network for image classification. The dataset consists of 70,000 grayscale images of 28x28 pixels each, with 10 different categories of clothing items.

## Installation
To run this project, you need to have Python installed along with the following libraries:
- TensorFlow
- Keras
- NumPy
- Matplotlib

You can install the required libraries using pip:
```bash
pip install tensorflow keras numpy matplotlib
```

## Usage
To train the model, run the following command:
```bash
python train_model.py
```

To evaluate the model, run:
```bash
python evaluate_model.py
```

## Dataset
The Fashion MNIST dataset can be downloaded from [here](https://github.com/zalandoresearch/fashion-mnist).

## Model Architecture
The neural network consists of the following layers:
- Input layer: 28x28 neurons (flattened)
- Dense layer: 128 neurons, ReLU activation
- Dropout layer: 0.2 dropout rate
- Dense layer: 10 neurons, softmax activation

## Results
The model achieves an accuracy of approximately 90% on the test dataset.

## License
This project is licensed under the MIT License.