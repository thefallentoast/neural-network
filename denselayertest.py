from layer import Dense
import numpy as np
from sklearn.model_selection import train_test_split

D = Dense(1)

D.build((2,))

X = np.array([[0.0105, 0.3129],
 [0.5564, 0.3484],
 [0.2220, 0.9394],
 [0.0684, 0.3678],
 [0.4088, 0.5920],
 [0.8367, 0.5142],
 [0.4668, 0.0524],
 [0.7530, 0.6791],
 [0.3554, 0.3600],
 [0.0600, 0.6155]])

y = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0])


Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=1337)

learning_rate = 0.00003

for i in range(70):
    print(Xtr.shape)
    output = D(Xtr[i])

    error = np.mean((output - ytr[i]) ** 2)

    if i % 10 == 0:
        print(f"Epoch {i}/70: error {error}")
    
    error_deriv = (2 / ytr.shape) * (output - ytr[i])

    output.back(error_deriv, learning_rate)
