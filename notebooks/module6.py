# # Module 6 - Deep Learning

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import keras

# Function to create model, required for KerasClassifier
def create_model(input_dim):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    # Compile model
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")


# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# create model
model = KerasClassifier(build_fn=create_model,
                        input_dim=2,
                        epochs=150,
                        batch_size=10,
                        verbose=0)
model.fit(df_train[["feature1", "feature2"]],
          df_train["class"])
model.fit()

# ## Deep Learning


# ## 
