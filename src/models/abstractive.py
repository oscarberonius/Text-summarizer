from __future__ import print_function

import math
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import json
from .ind_rnn import IndRNN
from os.path import dirname

class Abstractive:
    def __init__(self, data=None):
        data = data # list(dict({text, ingress}))
        self.batch_size = 64  # Batch size for training.
        self.epochs = 50  # Number of epochs to train for.
        self.latent_dim = 256  # Latent dimensionality of the encoding space.
        self.num_samples = 200000  # Number of samples to train on.
        if len(data) < self.num_samples:
            self.num_samples = len(data)
        print(self.num_samples)
        self.data_chunk_size = self.batch_size * 10 # Size of the data chunks that will be generated and trained on at a time. IMPORTANT: If too large then all RAM will be eaten.

        self.input_characters = set()
        self.target_characters = set()
        self.input_texts = []
        self.target_texts = []

        for dictionary in data[0:self.num_samples-1]:

            self.input_texts.append(dictionary['text'])
            self.target_texts.append(dictionary['ingress'])
            for c in dictionary['text']:
                self.input_characters.add(c)
            for c in dictionary['ingress']:
                self.target_characters.add(c)

        del data
        self.input_characters = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))
        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])


        print('Number of samples:', len(self.input_texts))
        print('Number of unique input tokens:', self.num_encoder_tokens)
        print('Number of unique output tokens:', self.num_decoder_tokens)
        print('Max sequence length for inputs:', self.max_encoder_seq_length)
        print('Max sequence length for outputs:', self.max_decoder_seq_length)

        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(self.target_characters)])

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(self.max_encoder_seq_length, self.num_encoder_tokens))
        encoder1 = IndRNN(self.latent_dim, return_state=True, return_sequences=True)
        encoder1_outputs, encoder1_states = encoder1(encoder_inputs)
        encoder2 = IndRNN(self.latent_dim, return_state=True)
        encoder_outputs, encoder_states = encoder2(encoder1_outputs)
        # We discard `encoder_outputs` and only keep the states.

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))#max_decoder_seq_length, num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_indrnn = IndRNN(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_indrnn(decoder_inputs,
                                            initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        chunks = int(math.floor(self.num_samples/self.data_chunk_size))
        residual_chunk = self.num_samples%self.data_chunk_size # TODO: Use this in a neat way

        for i in range(0,chunks):
            print("Training on chunk # ´", i)

            encoder_input_data, decoder_input_data, decoder_target_data = self.getTrainingChunk(i)

            model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=0.2)

        encoder_input_data, _, _ = self.getTrainingChunk(0) # Only for testing a sentence from the input when decoding later. TODO: Make proper decoding module 

        # Save model
        model.save('s2s.h5')

        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input = Input(shape=(self.latent_dim,))
        decoder_outputs, decoder_state = decoder_indrnn(
            decoder_inputs, initial_state=decoder_state_input)
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + [decoder_state_input],
            [decoder_outputs] + [decoder_state])

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())


        def decode_sequence(input_seq):
            # Encode the input as state vectors.
            states_value = encoder_model.predict(input_seq)

            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            # Populate the first character of target sequence with the start character.
            target_seq[0, 0, self.target_token_index['\t']] = 1.

            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:
                output_tokens, state = decoder_model.predict(
                    [target_seq] + [states_value])

                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = reverse_target_char_index[sampled_token_index]
                decoded_sentence += sampled_char

                # Exit condition: either hit max length
                # or find stop character.
                if (sampled_char == '\n' or
                len(decoded_sentence) > self.max_decoder_seq_length):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, self.num_decoder_tokens))
                target_seq[0, 0, sampled_token_index] = 1.

                # Update states
                states_value = state#[h, c]

            return decoded_sentence


        for seq_index in range(100):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            input_seq = encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = decode_sequence(input_seq)
            print('-')
            print('Input sentence:', self.input_texts[seq_index])
            print('Decoded sentence:', decoded_sentence)


    def getTrainingChunk(self, chunk):

        encoder_input_data = np.zeros(
            (self.data_chunk_size, self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='int8')
        decoder_input_data = np.zeros(
            (self.data_chunk_size, self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='int8')
        decoder_target_data = np.zeros(
            (self.data_chunk_size, self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='int8')

        for i in range(0, self.data_chunk_size):
            input_text = self.input_texts[i+self.data_chunk_size*chunk]
            target_text = self.target_texts[i+self.data_chunk_size*chunk]

            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.
        
        return encoder_input_data, decoder_input_data, decoder_target_data

    def reword(self, text):
        pass