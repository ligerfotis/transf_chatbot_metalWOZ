from __future__ import absolute_import, division, print_function
import csv
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import re
import pandas as pd
import string
import numpy as np
import yaml

tf.random.set_seed(1234)

with open('config.yaml') as f:
    prepare_config = yaml.load(f, Loader=yaml.FullLoader)["prepare_data"]
# Maximum number of samples to preprocess
MAX_SAMPLES = int(prepare_config["MAX_SAMPLES"])
vocabulary_size = int(prepare_config["vocabulary_size"])
data_filename = prepare_config["data_filename"]

#path_to_zip = tf.keras.utils.get_file( 'cornell_movie_dialogs.zip', origin= 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip', extract=True)
#path_to_dataset = os.path.join(os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

#path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
#path_to_movie_conversations = os.path.join(path_to_dataset, 'movie_conversations.txt')


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'([" "])+', " ", sentence)
    sentence = sentence.strip()
    # adding a start and an end token to the sentence
    return sentence


def load_conversations_Cornell(self, ):
    # dictionary of line id to text
    id2line = {}
    with open(self.path_to_movie_lines, errors='ignore') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]

    inputs, outputs = [], []
    with open(self.path_to_movie_conversations, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conversation) - 1):
            inputs.append(self.preprocess_sentence(id2line[conversation[i]]))
            outputs.append(self.preprocess_sentence(id2line[conversation[i + 1]]))
            if len(inputs) >= self.MAX_SAMPLES:
                return inputs, outputs
    return inputs, outputs


def load_conversations(filename):
    df = pd.read_csv(filename)
    inputs, outputs = [], []

    # remove empty sentense pair
    df = df.dropna()

    # Take a look at the first few rows
    string.punctuation = '"#$%&\'()*+-/:<=>@[\\]^_`{|}~'
    df.user = df.user.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
    df.system = df.system.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))

    user = df.user.tolist()
    system = df.system.tolist()

    for i in range(len(user)):
        inputs.append(preprocess_sentence(user[i]))
        outputs.append(preprocess_sentence(system[i]))

        if len(inputs) >= MAX_SAMPLES:
            return inputs, outputs

    return inputs, outputs


def split_conversations_metalwoz():
    filename = "data/metalwoz_full.csv"
    df = pd.read_csv(filename)
    inputs_train, inputs_test, outputs_train, outputs_test = [], [], [], []

    # remove empty sentense pair
    df = df.dropna()
    msk = np.random.rand(len(df)) < 0.8

    train = df[msk]
    test = df[~msk]

    user_train = train.user.tolist()
    system_train = train.system.tolist()
    user_test = test.user.tolist()
    system_test = test.system.tolist()

    print("Train Samples: " + str(len(user_train)) + ", " + str(len(system_train)))
    print("Test Samples: " + str(len(user_test)) + ", " + str(len(system_test)))

    for i in range(len(user_train)):
        if len(inputs_train) <= MAX_SAMPLES:
            inputs_train.append(preprocess_sentence(user_train[i]))
            outputs_train.append(preprocess_sentence(system_train[i]))
        else:
            break

    for i in range(len(user_test)):
        if len(inputs_test) <= MAX_SAMPLES:
            inputs_test.append(preprocess_sentence(user_test[i]))
            outputs_test.append(preprocess_sentence(system_test[i]))
        else:
            break

    with open("metalwoz_data/metalwoz_train.csv", "w") as train_file:
        csv_writer = csv.writer(train_file, delimiter=',')
        csv_writer.writerow(['user', 'system'])
        for q, a in zip(inputs_train, outputs_train):
            csv_writer.writerow([q, a])

    with open("metalwoz_data/metalwoz_test.csv", "w") as test_file:
        csv_writer = csv.writer(test_file, delimiter=',')
        csv_writer.writerow(['user', 'system'])
        for q, a in zip(inputs_test, outputs_test):
            csv_writer.writerow([q, a])


# Tokenize, filter and pad sentences

def tokenize_and_filter(inputs, outputs, START_TOKEN, END_TOKEN, tokenizer, MAX_LENGTH):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        # check tokenized sentence max length
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


def getDataset(MAX_LENGTH, BUFFER_SIZE, BATCH_SIZE):
    with open('config.yaml') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)["train"]

    resume_auto = bool(train_config["resume_auto"])
    resume_manual = bool(train_config["resume_manual"])
    manual_vocab_filename = train_config["manual_vocab_filename"]
    language = train_config["language"].lower()

    filename = "data/" + language + "/" + data_filename
    questions, answers = load_conversations(filename)

    # Build tokenizer using tfds for both questions and answers
    vocab_filename = "vocab_" + language + ".txt"
    if resume_auto and os.path.isfile(vocab_filename+".subwords"):
        tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_filename)

    elif resume_manual and os.path.isfile(manual_vocab_filename):
        tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(manual_vocab_filename)

    else:
        tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            questions + answers, target_vocab_size=vocabulary_size)
        vocab_filename = "vocab_" + language + ".txt"
        tokenizer.save_to_file(vocab_filename)

    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2
    print('Tokenized sample question: {}'.format(tokenizer.encode(questions[20])))

    questions, answers = tokenize_and_filter(questions, answers, START_TOKEN, END_TOKEN, tokenizer,
                                             MAX_LENGTH)
    print('Vocab size: {}'.format(VOCAB_SIZE))
    print('Number of samples: {}'.format(len(questions)))

    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions,
            'dec_inputs': answers[:, :-1]
        },
        {
            'outputs': answers[:, 1:]
        },
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, VOCAB_SIZE, tokenizer, START_TOKEN, END_TOKEN



