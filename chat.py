from __future__ import absolute_import, division, print_function, unicode_literals
__author__ = "Lygerakis Fotios"

import tensorflow as tf
import tensorflow_datasets as tfds
import yaml
from prepare_data import preprocess_sentence
from model import TransformerModel, load_embeddings
from util import loadCheckpoint_chat
import logging

__author__ = "Lygerakis Fotios"


logging.getLogger("tensorflow").setLevel(logging.ERROR)

with open('config.yaml') as f:
    chat_config = yaml.load(f, Loader=yaml.FullLoader)["chat"]
MAX_LENGTH = int(chat_config["MAX_LENGTH"])
BATCH_SIZE = int(chat_config["BATCH_SIZE"])
BUFFER_SIZE = int(chat_config["BUFFER_SIZE"])
language = chat_config["language"]


def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    vocab_filename = "vocab_" + language + ".txt"
    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_filename)
    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2
    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    emb_matrix = load_embeddings(vocab_size=VOCAB_SIZE, tokenizer=tokenizer, language=language)
    Transformer = TransformerModel(max_length=MAX_LENGTH, vocab_size=VOCAB_SIZE, embedding_matrix=emb_matrix)

    sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)
    output = tf.expand_dims(START_TOKEN, 0)

    # Create a new basic model instance
    model = Transformer.model

    checkpoint_path = loadCheckpoint_chat(VOCAB_SIZE)
    try:
        model.load_weights(checkpoint_path)
        print("Model loaded from checkpoint " + checkpoint_path + "Loaded")
    except ValueError:
        print("Error loading checkpoint " + checkpoint_path)
        print("ValueError:" + str(ValueError))

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), tokenizer


def predict(sentence):
    prediction, tokenizer = evaluate(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence


while 1:
    question = input("Question: ")
    print("")
    output = predict(question)
