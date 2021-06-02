# # Module 7 - Deep Learning 

import keras


# NLP


# RNN 

# Function to create model, required for KerasClassifier

# Create and RNN model
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
