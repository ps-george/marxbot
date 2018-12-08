'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.  
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, TimeDistributed, GRU
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import boto3
import numpy as np
import random
import sys
import io
import json

client = boto3.client('s3')
s3 = boto3.resource('s3')
path = get_file( 'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of seqlen characters
seqlen = 40
VOCAB_SIZE = len(chars)
step = 5
sentences = []
next_chars = []
for i in range(0, len(text) - seqlen, step):
    sentences.append(text[i: i + seqlen])
    next_chars.append(text[i + seqlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
#  Create input and target sequences
x = np.zeros((len(sentences), seqlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
hidden_dim = 128
num_layers = 3
model = Sequential()
model.add(GRU(hidden_dim, input_shape=(seqlen, VOCAB_SIZE)))
model.add(Dense(VOCAB_SIZE, activation='softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(epoch, length=400):
    ENDINGS = ['.', '!', '?', ' ']
    start_index = random.randint(0, len(text) - seqlen - 1)
    generated = {}
    next_char = ''
    for diversity in [0.4, 0.5, 0.6, 0.7]:
        generated[str(diversity)] = ''
        sentence = text[start_index: start_index + seqlen]
        generated[str(diversity)] += sentence
        i = 0
        while(i<length):
            prev_char = next_char
            x_pred = np.zeros((1, seqlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            if next_char == '\n' and prev_char not in ENDINGS:
                next_char = np.random.choice([' ',',',';'])

            # end on punctuation
            if i == length-1:
                if next_char not in ENDINGS:
                    i -= 1
                if next_char == ' ':
                    next_char = '.'
            generated[str(diversity)] += next_char
            sentence = sentence[1:] + next_char
            i+=1
    return generated

def store_weights(model, epoch):
    local_params_path = "model_params" # temp path to export your network parameters i.e. weights
    bucket_name = "psgeorge-deeplearning-bucket" # s3 key to save your network to
    s3_params_key = "v2_model_params_{}_epochs" # s3 key to save your network parameters i.e. weights
    model.save_weights(local_params_path.format(epoch))
    s3.Bucket(bucket_name).upload_file(local_params_path, s3_params_key.format(epoch))
    return


def main():
    BATCH_SIZE = 128
    EPOCHS = 30
    nb_epoch = 0
    train=True

    if train:
        while nb_epoch < 60:
            nb_epoch += 1
            model.fit(x, y,
                  batch_size=BATCH_SIZE,
                  epochs=1)
            txt = generate_text(nb_epoch)
            for k,v in txt.items():
                print('----- Diversity = {} -----'.format(k))
                print(v)
                print('\n')
            if nb_epoch % 10 == 0:
                # Save/log generated text somewhere, maybe just for checkpoint epochs
                store_weights(model)
                filename = 'output_epoch_{}.txt'.format(nb_epoch)
                with open(filename, 'a') as f:
                    json.dump(txt, f)
                s3.Bucket(bucket_name).upload_file(filename, filename)
    else:
        model.load_weights("model_params.h5")
        for i in range (5):
            txt = generate_text(None, 500)
            for k,v in txt.items():
                print('----- Diversity = {} -----'.format(k))
                print(v)
                print('\n')

if __name__=="__main__":
    main()
