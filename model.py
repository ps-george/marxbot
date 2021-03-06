from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
import io
import json

class ChatBot(object):
    def __init__(self, sources=['data.txt'], sequence_length=100, diversity=0.7, s3bucket=''):
        self.__model = None
        self.__diversity = diversity
        self.__s3bucket = s3bucket
        self.__sources = sources 
        self.__seqlen = sequence_length
        self.__sentences = []
        self.__next_chars = []
        self.__ci = {}
        self.__ic = {}
        self.__text = ''
        self.__vocab_size = None
        self.__chars = None
        self.__corpus_length = None
        self.__x = None
        self.__y = None
        self.build_vocabulary()
        self.build_model()
        pass

    def build_vocabulary(self):
        """
        Build vocabulary of characters from the provided text files
        """

        print('Building vocabulary...')
        for line in self.text_gen():
            self.__text += line

            # for word embedding
            # for word in line:
            #    self.__vocab.add(word)
        # using character level model for now
        self.__chars = sorted(list(set(self.__text)))
        self.__vocab_size = len(self.__chars)
        self.__ci = dict((c, i) for i, c in enumerate(self.__chars))
        self.__ic = dict((i, c) for i, c in enumerate(self.__chars))
        self.__corpus_length = len(self.__text)
        print('Corpus length:', self.__corpus_length)
        return

    def clean_line(self, line):
        """Remove bad punctuation and numerics from the text."""
        BAD_PUNCTUATION = ['(', ')','`','/', '{', '}', '*', '%', '$', '>', '=','_', '\\', '[', ']', '\x1f'] 
        # PUNCTUATION = ['!','?', ':', ';', ',', '.','-', '"', '\'',] # for this model these are fine
        NUMERICS = [str(x) for x in range(10)]
        CHECKLIST = BAD_PUNCTUATION + NUMERICS
        # Return the line as list of words without punctuation and numeric symbols
        result = []
        line = line.replace('\xef', ' ')
        for char in CHECKLIST:
            line = line.replace(char, '')
            line = line.lower()
        return line

    def text_gen(self):
        """Generate cleaned text from the sources."""
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)
       
    def vectorization(self):
        """Create input and target sequences for the RNN."""
        nb_sequences = len(self.__sentences)
        self.__x = np.zeros((nb_sequences, self.__seqlen, self.__vocab_size), dtype=np.bool)
        self.__y = np.zeros((nb_sequences, self.__vocab_size), dtype=np.bool)
        for i, sentence in enumerate(self.__sentences):
            for t, char in enumerate(sentence):
                self.__x[i, t, self.__ci[char]] = 1
            self.__y[i, self.__ci[self.__next_chars[i]]] = 1
        return

    def cut_text(self, step):
        """Cut the text into sentences `seqlen` long and store the next characters following a sequence."""
        for i in range(0, self.__corpus_length - self.__seqlen, step):
            self.__sentences.append(self.__text[i: i + self.__seqlen])
            self.__next_chars.append(self.__text[i + self.__seqlen])
        
    def build_model(self):
        """Build the RNN model; using one LSTM and a Dense activation layer."""
        print('Build model...')
        hidden_dim = 128
        self.__model = Sequential()
        self.__model.add(LSTM(hidden_dim, input_shape=(self.__seqlen, self.__vocab_size)))
        self.__model.add(Dense(self.__vocab_size, activation='softmax'))
        optimizer = RMSprop(lr=0.01)
        self.__model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def save_to_s3(self, local_name, s3_name):
        """Used when training on AWS servers. Save file to s3 bucket."""
        import boto3
        # save to file
        BUCKET_NAME = self.__s3bucket # s3 key to save your network to
        s3 = boto3.resource('s3')
        s3.Bucket(BUCKET_NAME).upload_file(local_name, s3_name)
        return

    def train_online(self):
        self.train(True)
        import boto3
        # When training is finished, terminate this instance to save money
        ec2 = boto3.resource('ec2', region_name='eu-central-1')
        ec2.instances.filter(InstanceIds=['i-04fc019e944e13688']).terminate()

    def train(self, online=False):
        """Train the neural network"""
        self.cut_text(1)
        nb_sequences = len(self.__sentences)
        print('nb sequences:', len(self.__sentences))

        # Create input and target sequences
        print('Vectorization...')
        self.vectorization()

        #params
        BATCH_SIZE = 128
        EPOCHS = 31
        nb_epoch = 0

        if online:
            # Test save functions before training
            weights_filename = "movie_model_params_{}_epochs".format(nb_epoch) # temp path to export your network parameters i.e. weights
            self.__model.save_weights(weights_filename)
            self.save_to_s3(weights_filename, weights_filename)

        # Training loop
        while nb_epoch < EPOCHS:
            nb_epoch += 1
            self.__model.fit(self.__x, self.__y,
                batch_size=BATCH_SIZE,
                epochs=1)
            txt = self.generate_text()
            for k,v in txt.items():
                print('----- Diversity = {} -----'.format(k))
                print(v)
                print('\n')
            if nb_epoch % 5 == 0 or nb_epoch == 1:
                weights_filename = "movie_model_params_{}_epochs".format(nb_epoch) # temp path to export your network parameters i.e. weights
                self.__model.save_weights(weights_filename)
                # Log generated text
                filename = 'movie_output_epoch_{}.txt'.format(nb_epoch)
                with open(filename, 'a') as f:
                    json.dump(txt, f)
                if online:
                    self.save_to_s3(weights_filename, weights_filename)
                    self.save_to_s3(filename, filename)

    def sample(self, preds, temperature=1.0):
        """helper function to sample an index from a probability array."""
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def respond(self, seed):
        """Generate text based on a given seed text."""
        ENDINGS = ['.', '!', '?', '\n', ',', ';']
        generated = {}
        seed = seed[-self.__seqlen:]
        next_char = ''
        sentence = seed
        generated= ''
        while (next_char not in ENDINGS):
            prev_char = next_char
            x_pred = np.zeros((1, len(sentence), self.__vocab_size))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.__ci[char]] = 1.
            preds = self.__model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, self.__diversity)
            next_char = self.__ic[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char
        if generated[-1] in [',',';']:
            generated = generated[:-1]
        return generated

    def generate_text(self, length=400):
        """Generate text using the trained model."""
        ENDINGS = ['.', '!', '?', ' ']
        start_index = random.randint(0, self.__corpus_length - self.__seqlen - 1)
        generated = {}
        next_char = ''
        for diversity in [0.4, 0.5, 0.6, 0.7]:
            generated[str(diversity)] = ''
            sentence = self.__text[start_index: start_index + self.__seqlen]
            generated[str(diversity)] += sentence
            i = 0
            while(i<length):
                prev_char = next_char
                x_pred = np.zeros((1, self.__seqlen, self.__vocab_size))
                for t, char in enumerate(sentence):
                    x_pred[0, t, self.__ci[char]] = 1.

                preds = self.__model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = self.__ic[next_index]
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

    def load(self, weights_file):
        self.__model.load_weights(weights_file)

def main():
    chatbot = ChatBot(s3bucket='')
    # either train, or load weights
    chatbot.train_online()

    # chatbot.load('model_params.h5')
    # for i in range (5):
    #     txt = chatbot.generate_text(None, 500)
    #     for k,v in txt.items():
    #         print('----- Diversity = {} -----'.format(k))
    #         print(v)
    #         print('\n')


if __name__=="__main__":
    main()
