# deep-learning-challenge

A neural network machine learning project on applicants for charity funding. We wish to predict the success of an applicant (boolean) on the basis of their profile:

```
APPLICATION_TYPE        Alphabet Soup application type
AFFILIATION             Affiliated sector of industry
CLASSIFICATION          Government organisation classification
USE_CASE                Use case for funding
ORGANIZATION            Organisation type
STATUS                  Active status
INCOME_AMT              Income classification
SPECIAL_CONSIDERATIONS  Special considerations for application
ASK_AMT                 Funding amount requested
```


## Overview

### Feature engineering 

Features were one-hot encoded. This required bucketing low-frequency instances of `APPLICATION_TYPE` and `CLASSIFICATION`. The model was further optimised by identifying the log-normal distribution of `ASK_AMT`, with a significant spike at 5000, which was extracted to a separate label.

The initial model was trained with two hidden layers, using the rectified linear unit (ReLU) activation function, using a 32-16-1 architecture and a 20% training set.

### Initial results

The initial model achieved an accuracy of 0.735 and loss of 0.5617

### Optimisation

A 3x3x3x3 grid search approach was used to evaluate the model across the following hyperparameters:

learning rates: `0.001, 0.01, 0.1`

batch sizes: `32, 64, 128`

layer architectures (number of units per layer): `(64, 32), (128, 64), (256, 128)`

activation functions: `'relu', 'tanh', 'sigmoid'`

The significantly increased number of units per layers is moderated somewhat by the inclusion of a 50% dropout.

Testing all permutations required generating models across 3^4, or 81 permutations, which was achieved via nesting loops across each list of parameter settings. A 4-fold cross-validation approach was also implemented in this search. Due to the length of this process, an early stopping function was implemented. This meant that even though 100 epochs were specified, training would stop if 10 epochs passed without improvement to the accuracy.

The legacy Adam optimiser was chosen due to better performance on ARM64 processors.



```
Iteration 1/81: Learning Rate = 0.001, Batch Size = 32, Layer Config = (32, 16), Activation = relu
Iteration 2/81: Learning Rate = 0.001, Batch Size = 32, Layer Config = (32, 16), Activation = tanh
Iteration 3/81: Learning Rate = 0.001, Batch Size = 32, Layer Config = (32, 16), Activation = sigmoid
Iteration 4/81: Learning Rate = 0.001, Batch Size = 32, Layer Config = (64, 32), Activation = relu
Iteration 5/81: Learning Rate = 0.001, Batch Size = 32, Layer Config = (64, 32), Activation = tanh
Iteration 6/81: Learning Rate = 0.001, Batch Size = 32, Layer Config = (64, 32), Activation = sigmoid
Iteration 7/81: Learning Rate = 0.001, Batch Size = 32, Layer Config = (128, 64), Activation = relu
Iteration 8/81: Learning Rate = 0.001, Batch Size = 32, Layer Config = (128, 64), Activation = tanh
Iteration 9/81: Learning Rate = 0.001, Batch Size = 32, Layer Config = (128, 64), Activation = sigmoid
```


## Result of tuning


```
Best Hyperparameters
    {
        'learning_rate': 0.01, 
        'batch_size': 128, 
        'layer_config': (64, 32), 
        'activation_func': 'sigmoid'
    }


Best Validation Accuracy
    0.7317123860120773


Best Hyperparameters
    {
        'learning_rate': 0.001, 
        'batch_size': 32, 
        'layer_config': (64, 32), 
        'activation_func': 
        'relu'
    }

Best Validation Accuracy
    0.7324412167072296


Best Hyperparameters:
    {
        'learning_rate': 0.002,
         'batch_size': 30,
         'layer_config': (16, 16),
         'activation_func': 'swish'
    }

Best Validation Accuracy:
    0.7318290024995804

```



### Disclaimer

No external code or other content has been used in this project, except where specifically provided in the assignment resources. LLMs used for model iteration and refinement.
