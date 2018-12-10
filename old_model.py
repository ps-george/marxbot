import numpy as np
from string import ascii_letters
from keras import Sequential
from keras.layers import Activation, LSTM, Dense, TimeDistributed


def main():
    # Prepare the training data
    text = open("Plato/The-Republic.htm").read()

    # Remove html tags
    tags = ["<p>", "</p>"]
    hopen = ["<h{}>".format(num) for num in range(1,4)]
    hclose = ["</h{}>".format(num) for num in range(1,4)]
    tags += hopen + hclose
    for tag in tags:
        text = text.replace(tag, "")
    text = text.replace("&mdash;", "-")
    text = text.replace("<", "")
    text = text.replace(">", "")
    print(len(text.split(" ")))

    # using character generation model
    chars = list(set(text))
    print(chars)
    VOCAB_SIZE = len(chars)
    ix_to_char = {ix:char for ix, char in enumerate(chars)}
    char_to_ix = {char:ix for ix, char in enumerate(chars)}

    # HYPERPARAMETERS
    SEQ_LENGTH = 30
    HIDDEN_DIM = 100
    LAYER_NUM = 4
    GENERATE_LENGTH = 500
    BATCH_SIZE = 32

    #  Create input and target sequences
    sequences = len(text)//SEQ_LENGTH
    X = np.zeros((sequences, SEQ_LENGTH, VOCAB_SIZE))
    y = np.zeros((sequences, SEQ_LENGTH, VOCAB_SIZE))
    for i in range(0, sequences):
        X_sequence = text[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
        X_sequence_ix = [char_to_ix[value] for value in X_sequence]
        input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
        for j in range(SEQ_LENGTH):
            input_sequence[j][X_sequence_ix[j]] = 1
        X[i] = input_sequence

        y_sequence = text[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
        for j in range(SEQ_LENGTH):
            target_sequence[j][y_sequence_ix[j]] = 1
        y[i] = target_sequence

    # Create sequential deep model
    model = Sequential()
    model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
    for i in range(LAYER_NUM - 1):
        model.add(LSTM(HIDDEN_DIM, return_sequences=True))
        model.add(TimeDistributed(Dense(VOCAB_SIZE)))
        model.add(Activation('softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    
    def generate_text(model, length, seed=None):
        ENDINGS = ['.', '!', '?']
        # Seed text generation with a number characters
        if seed:
            # 
            pass

        # Length parameter can be random, but always end on a punctionation mark
        ix = [char_to_ix['W']]
        y_char = [ix_to_char[ix[-1]]]
        X = np.zeros((1, length, VOCAB_SIZE))
        i = 0
        while(i<length):
            X[0, i, :][ix[-1]] = 1
            ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
            y_char.append(ix_to_char[ix[-1]])
            i += 1
        return ('').join(y_char)

    print(generate_text(model, GENERATE_LENGTH))
    # Train model
    nb_epoch = 0
    while True:
        print('\n\n')
        model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=1)
        nb_epoch += 1
        print(generate_text(model, GENERATE_LENGTH))
        if nb_epoch % 10 == 0:
            model.save_weights('checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, nb_epoch))


if __name__=="__main__":
    main()
