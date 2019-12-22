## Overview
Matrix Factorization with gradient descent (template)

### Development Setup
1. Install pyenv
    ```
    $ brew install pyenv
    ```

2. Install python version 3.7.4
    ```
    $ pyenv install 3.7.4
    ```

3. Initialize pyenv
    ```
    $ pyenv init
    ```

4. Make sure you are using the right python version (3.7.4). You may want to add the following to your bash settings:
    ```
    # Loads pyenv automatically
    eval "$(pyenv init -)"

    # Prevents creation of pycache files
    export PYTHONDONTWRITEBYTECODE=true
    ```

5. Upgrade pip
    ```
    $ pip install --upgrade pip
    ```

6. Install dependencies
    ```
    $ pip install -r requirements.txt
    ```

7. Run the code
    ```
    $ python app.py
    ```

### Sample Output

    ------------------- INPUT --------------------
    [[5 3 0 1]
     [4 0 0 1]
     [1 1 0 5]
     [1 0 0 4]
     [0 1 5 4]]

    --------------- HYPERPARAMETERS ----------------

    K: 2
    alpha (learning rate): 0.05
    beta (regularization): 0.01
    target_accuracy: 0.99
    max_iterations: 1000
    tol: 0.0001

    ------------------ TRAINING --------------------
    Number of training samples: 13

    Iteration: 10 ; error = 0.6953
    Iteration: 20 ; error = 0.1394
    Iteration: 30 ; error = 0.0659
    Iteration: 40 ; error = 0.0351
    Iteration: 50 ; error = 0.0216
    Iteration: 60 ; error = 0.0152
    Iteration: 70 ; error = 0.0129
    Iteration: 80 ; error = 0.0116
    Iteration: 90 ; error = 0.0112
    Iteration: 97 ; error = 0.0114
    Target error difference (tol) 0.0001 reached.
    Training time: 0.0283s

    ------------------ RESULTS ---------------------

    Full matrix:
    [[4.98410279 2.99956407 3.36916102 1.01299501]
     [3.99438678 2.42353852 2.04634334 1.01090409]
     [1.00989163 1.0039021  5.03071931 4.98151913]
     [1.01003283 0.79970579 4.07342854 3.99419062]
     [1.46899817 1.01864929 4.98750062 3.99674531]]

    -------------- TRAINED PARAMETERS --------------

    Bias:
    2.769230769230769

    Item bias:
    [-0.04812197 -0.99631352  1.06112858  0.03967185]

    User bias:
    [ 0.11894482 -0.24765476  0.30930861 -0.18215793 -0.04182148]

    Item latent matrix:
    [[-0.36340674  1.34937584]
     [ 0.91397388  1.0049598 ]
     [-0.01623151 -1.28625188]
     [ 0.24713568 -0.96272711]
     [-0.70271012 -0.79869938]]

    User latent matrix:
    [[-0.06404323  1.57167141]
     [ 0.06188038  0.83756483]
     [-0.93218956 -0.68098678]
     [-0.1048773  -1.44731029]]
