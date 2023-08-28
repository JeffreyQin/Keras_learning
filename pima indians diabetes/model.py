from numpy import loadtxt;
from tensorflow import keras;
from keras.models import Sequential;
from keras.layers import Dense;


# loading data

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

# construct model

model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile model

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# train model

model.fit(
    X,
    Y,
    epochs=150,
    batch_size=10
)

# evaluate model
_, accuracy = model.evaluate(X, Y)