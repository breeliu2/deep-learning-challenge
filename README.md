# deep-learning-challenge

# Introduction
The aim of this analysis was to assist nonprofit foundation Alphabet Soup in selecting applicants for funding with the highest likelihood of success in their ventures. Leveraging my expertise in machine learning and neural networks, I developed a binary classifier using the provided dataset's features to predict the potential success of applicants if funded by Alphabet Soup.


# Data Preprocessing
## What variable(s) are the target(s) for your model?
* The 'IS_SUCCESSFUL' column in the 'application_df' dataset serves as the target variable for our prediction task. Our goal is to predict whether the money was used effectively, as indicated by the binary values in this column.

## What variable(s) are the features of your model? 
* The feature variables we used are:
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested

## What variable(s) should be removed from the input data because they are neither targets nor features?
* Identification columns: "EIN" and "NAME" columns are identification columns that typically provide unique identifiers for each organization. It is often a good practice to drop identification columns before training a machine learning model. These columns do not contribute to the model's predictive power, and keeping them might lead to overfitting. 


# Compiling, Training, and Evaluating the Model

## Baseline model 
This model displays two hidden layers (80 and 30 neurons, respectively) and an output layer with a single neuron. The 'relu' activation function is used in the hidden layers to introduce non-linearity, and the 'sigmoid' activation function is used in the output layer for binary classification. 

<img width="587" alt="Screenshot 2023-08-04 at 12 55 51 AM" src="https://github.com/breeliu2/deep-learning-challenge/assets/124847109/71e430dc-2536-4891-9a62-275bbce93901">

This model did not reach the target performance
 

<img width="555" alt="Screenshot 2023-08-04 at 12 56 23 AM" src="https://github.com/breeliu2/deep-learning-challenge/assets/124847109/5da3584b-fb7d-4e07-9366-b5d336f5d1b2">


## Optimization Model 1 
Model Architecture:

The model consists of three layers: two hidden layers and one output layer.
The first hidden layer has 80 neurons, and the second hidden layer has 30 neurons.
The output layer has a single neuron, suitable for binary classification tasks.

Activation Functions:
The first and second hidden layers use the 'tanh' (hyperbolic tangent) activation function, which introduces non-linearity.
The output layer uses the 'sigmoid' activation function, suitable for binary classification tasks, to produce a probability value between 0 and 1.

Dropout Layers:
Dropout layers with a dropout rate of 50% (0.5) are added after each hidden layer.
Dropout is used to help prevent overfitting by randomly deactivating neurons during training.

Input Features:
The number of input features is determined by the length of the input data X_train_scaled[0].

<img width="783" alt="Screenshot 2023-08-04 at 12 42 10 AM" src="https://github.com/breeliu2/deep-learning-challenge/assets/124847109/5e539a39-f373-4f2d-9e37-a6f5d15864f3">

This model did not reach the target performance

<img width="564" alt="Screenshot 2023-08-04 at 12 53 23 AM" src="https://github.com/breeliu2/deep-learning-challenge/assets/124847109/b93cdbb7-dd57-4c6f-a960-2e14d33e8aaa">

## Optimization Model 2
Model Architecture:
The model consists of three hidden layers and one output layer.
Number of Neurons (Units) in Each Layer:
units_1 = 80: The first hidden layer has 80 neurons.
units_2 = 30: The second hidden layer has 30 neurons.
units_3 = 20: The third hidden layer has 20 neurons.
units=1 in the output layer, which has a single neuron for binary classification.

Input Features:
The number of input features is determined using len(X_train_scaled[0]).

Sequential Model:
The code initializes a sequential model using tf.keras.models.Sequential().

Adding Hidden Layers:
The first hidden layer is added to the model using nn.add().
The second and third hidden layers are added similarly.
The 'relu' (Rectified Linear Unit) activation function is used for all the hidden layers, introducing non-linearity.

Output Layer:
The output layer is added to the model using nn.add().
It has a single neuron (1 unit) since this is a binary classification problem.
The 'sigmoid' activation function is used in the output layer to produce a probability value between 0 and 1.

<img width="752" alt="Screenshot 2023-08-04 at 1 00 02 AM" src="https://github.com/breeliu2/deep-learning-challenge/assets/124847109/6f1c618b-fd6f-445d-804c-17f16aec2ed6">

The model did not reach the target performance: 

<img width="573" alt="Screenshot 2023-08-04 at 1 04 43 AM" src="https://github.com/breeliu2/deep-learning-challenge/assets/124847109/8689df35-fb0d-4fa9-aae5-d48f884a19b4">

## Optimization Model 3 
Model Architecture:
The model consists of two hidden layers and one output layer.

Number of Neurons (Units) in Each Layer:
units_1 = 80: The first hidden layer has 80 neurons.
units_2 = 30: The second hidden layer has 30 neurons.
units=1 in the output layer, which has a single neuron for binary classification.

Input Features:
The number of input features is determined using len(X_train_scaled[0]).

Sequential Model:
The code initializes a sequential model using tf.keras.models.Sequential().

Adding Hidden Layers:
The first hidden layer is added to the model using nn.add().
The second hidden layer is added similarly.
The 'relu' (Rectified Linear Unit) activation function is used for both hidden layers, introducing non-linearity.

Output Layer:
The output layer is added to the model using nn.add().
It has a single neuron (1 unit) since this is a binary classification problem.
The 'sigmoid' activation function is used in the output layer to produce a probability value between 0 and 1.


<img width="750" alt="Screenshot 2023-08-04 at 1 08 28 AM" src="https://github.com/breeliu2/deep-learning-challenge/assets/124847109/697ccf66-ae86-4311-a9cf-d918e8627c56">

This model did not reach the target performance

<img width="629" alt="Screenshot 2023-08-04 at 1 15 19 AM" src="https://github.com/breeliu2/deep-learning-challenge/assets/124847109/2cf91280-203e-4ab3-be79-a6dc830d1d10">


# Summary
Since none of the models were able to reach the desired performance, there are several strategies that I could use in future models to enhance the performance. These include increasing the dataset size, ensuring thorough data cleaning, exploring different algorithms, determining feature importance, handling bias and outliers, and implementing data binning techniques. Each of these steps aims to improve the model's capacity to capture crucial patterns while minimizing noise, ultimately leading to increased accuracy in the classification task. 
