import numpy as np
from keras import Sequential
from keras.layers import Activation, LSTM, Dense, TimeDistributed

def load_text(filename):
    text = """
The argument of the Republic is the search after Justice, the nature of which is first hinted at by Cephalus, the just and blameless old man&mdash;then
discussed on the basis of proverbial morality by Socrates and Polemarchus&mdash;then
caricatured by Thrasymachus and partially explained by Socrates&mdash;reduced
to an abstraction by Glaucon and Adeimantus, and having become invisible
in the individual reappears at length in the ideal State which is
constructed by Socrates. The first care of the rulers is to be education,
of which an outline is drawn after the old Hellenic model, providing only
for an improved religion and morality, and more simplicity in music and
gymnastic, a manlier strain of poetry, and greater harmony of the
individual and the State. We are thus led on to the conception of a higher
State, in which 'no man calls anything his own,' and in which there is
neither 'marrying nor giving in marriage,' and 'kings are philosophers'
and 'philosophers are kings;' and there is another and higher education,
intellectual as well as moral and religious, of science as well as of art,
and not of youth only but of the whole of life. Such a State is hardly to
be realized in this world and quickly degenerates. To the perfect ideal
succeeds the government of the soldier and the lover of honour, this again
declining into democracy, and democracy into tyranny, in an imaginary but
regular order having not much resemblance to the actual facts. When 'the
wheel has come full circle' we do not begin again with a new period of
human life; but we have passed from the best to the worst, and there we
end. The subject is then changed and the old quarrel of poetry and
philosophy which had been more lightly treated in the earlier books of the
Republic is now resumed and fought out to a conclusion. Poetry is
discovered to be an imitation thrice removed from the truth, and Homer, as
well as the dramatic poets, having been condemned as an imitator, is sent
into banishment along with them. And the idea of the State is supplemented
by the revelation of a future life.

"""
    return text


def main():
    # Prepare the training data
    text = load_text("plato.htm")

    # using character generation model
    chars = list(set(text))
    VOCAB_SIZE = len(chars)
    ix_to_char = {ix:char for ix, char in enumerate(chars)}
    char_to_ix = {char:ix for char, ix in enumerate(chars)}

    # HYPERPARAMETERS
    SEQ_LENGTH = 30
    HIDDEN_DIM = 100
    LAYER_NUM = 4

    #  Create input and target sequences
    X = np.zeros((len(text)/SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE))
    y = np.zeros((len(text)/SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE))
    for i in range(0, len(text)/SEQ_LENGTH):
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

        ix = [np.random.randint(VOCAB_SIZE)]
        y_char = [ix_to_char[ix[-1]]]
        X = np.zeros((1, length, VOCAB_SIZE))
        while(i<length or y_char[-1] in ENDINGS):
            X[0, i, :][ix[-1]] = 1
            print(ix_to_char[ix[-1]], end="")
            ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
            y_char.append(ix_to_char[ix[-1]])
            i+=1
        return ('').join(y_char)

    # Train model
    nb_epoch = 0
    while True:
        print('\n\n')
        model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
        nb_epoch += 1
        generate_text(model, GENERATE_LENGTH)
        if nb_epoch % 10 == 0:
            model.save_weights('checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, nb_epoch))


if __name__=="__main__":
    main()
