# Potato-Leaf-Disease-Classification

# Introduction
The Potato Disease Classification project aims to develop a deep learning model capable of identifying diseases in potato plant images. The dataset used for training, validation, and testing comes from the Plant Village dataset on Kaggle. The project involves preprocessing the data, building a Convolutional Neural Network (CNN) model using TensorFlow and Keras, training the model, and evaluating its performance.

# Dataset
The dataset consists of images of healthy and diseased potato plants, with various diseases such as early blight, late blight, and healthy leaves. The dataset is divided into three sets: training, validation, and testing. The images are preprocessed using data augmentation techniques such as rescaling, rotation, and horizontal flipping to increase the diversity of the training set.

# Model Architecture
The CNN model is built using TensorFlow and Keras. It consists of several convolutional layers followed by max-pooling layers for feature extraction. The final layers include fully connected layers with ReLU activation and a softmax output layer for classification. The model architecture is summarized as follows:

# Model Training
The model is compiled using the Adam optimizer, SparseCategoricalCrossentropy loss function, and accuracy as the metric. It is trained using the training set and validated on the validation set. The training process is visualized by plotting accuracy and loss curves over epochs.

# Model Evaluation
The trained model is evaluated on the test set to assess its generalization performance. The evaluation includes calculating loss and accuracy metrics.

# Results and Analysis
The project includes visualizations of the training and validation accuracy/loss curves. These visualizations provide insights into the model's performance during training. Additionally, a sample inference is demonstrated on a test image, showcasing the model's prediction compared to the actual label. A function for batch prediction is also provided, along with a visual representation of predictions on a few test images.

# Conclusion
The Potato Disease Classification project successfully develops and trains a deep learning model to classify diseases in potato plant images. The model demonstrates good performance on the test set, as indicated by accuracy and loss metrics. The provided visualizations offer insights into the training process and model behavior.
