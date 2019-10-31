from __future__ import absolute_import, division, print_function, unicode_literals
__author__ = "Lygerakis Fotios"

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from prepare_data import preprocess_sentence, getDataset
    from model import TransformerModel, load_embeddings, exp_decay_scedule, custom_scedule
    import os
    import yaml
    from util import loadCheckpoint_train, get_available_gpus, getCallbacks
    import logging


logging.getLogger("tensorflow").setLevel(logging.ERROR)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
tf.random.set_seed(1234)

with open('config.yaml') as f:
    train_config = yaml.load(f, Loader=yaml.FullLoader)["train"]

MAX_LENGTH = int(train_config["MAX_LENGTH"])
BATCH_SIZE = int(train_config["BATCH_SIZE"])
BUFFER_SIZE = int(train_config["BUFFER_SIZE"])
EPOCHS = int(train_config["EPOCHS"])
language = train_config["language"]


# utilizing multiple GPUs
num_gpus = get_available_gpus()
if len(num_gpus) > 0:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        [dataset_train, VOCAB_SIZE, tokenizer, START_TOKEN, END_TOKEN] = getDataset(MAX_LENGTH, BUFFER_SIZE, BATCH_SIZE)
        print(dataset_train)

        emb_matrix = load_embeddings(vocab_size=VOCAB_SIZE, tokenizer=tokenizer, language=language)

        Transformer = TransformerModel(max_length=MAX_LENGTH, vocab_size=VOCAB_SIZE, embedding_matrix=emb_matrix)
        model = Transformer.model
        model.compile(optimizer=Transformer.optimizer, loss=Transformer.loss_function, metrics=[Transformer.accuracy])
        # Retrieve Checkpoint if available
        checkpoint_path, model = loadCheckpoint_train(VOCAB_SIZE, language, model)
        cp_callback, tensorboard_callback, lr_callback = getCallbacks(checkpoint_path)

        model.fit(dataset_train, epochs=EPOCHS, callbacks=[cp_callback, tensorboard_callback, lr_callback],
                  verbose=1)
        model.save_weights(checkpoint_path.format(epoch=0))
else:
    [dataset_train, VOCAB_SIZE, tokenizer, START_TOKEN, END_TOKEN] = getDataset(MAX_LENGTH, BUFFER_SIZE, BATCH_SIZE)
    print(dataset_train)

    emb_matrix = load_embeddings(vocab_size=VOCAB_SIZE, tokenizer=tokenizer, language=language)

    Transformer = TransformerModel(max_length=MAX_LENGTH, vocab_size=VOCAB_SIZE, embedding_matrix=emb_matrix)
    model = Transformer.model
    model.compile(optimizer=Transformer.optimizer, loss=Transformer.loss_function, metrics=[Transformer.accuracy])
    # Retrieve Checkpoint if available
    checkpoint_path, model = loadCheckpoint_train(VOCAB_SIZE, language, model)
    cp_callback, tensorboard_callback, lr_callback = getCallbacks(checkpoint_path)

    model.fit(dataset_train, epochs=EPOCHS, callbacks=[cp_callback, tensorboard_callback, lr_callback], verbose=1)
    model.save_weights(checkpoint_path.format(epoch=0))




def evaluate(sentence):
    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    test_Transformer = TransformerModel(max_length=MAX_LENGTH, vocab_size=VOCAB_SIZE, embedding_matrix=emb_matrix)
    test_model = test_Transformer.model
    test_model.load_weights(checkpoint_path)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input
    output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict():
    sentences = ['Can you help me with my problem?', "Where should I eat today?", 'Is Spain a nice city?']
    for sentence in sentences:
        prediction = evaluate(sentence)
        predicted_sentence = tokenizer.decode(
            [i for i in prediction if i < tokenizer.vocab_size])

        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))

    prediction = evaluate(sentences[2])
    question = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])
    for _ in range(5):
        # feed the model with its previous output
        prediction = evaluate(question)
        predicted_sentence = tokenizer.decode(
            [i for i in prediction if i < tokenizer.vocab_size])

        print('Input: {}'.format(question))
        print('Output: {}'.format(predicted_sentence))
        print('')
        question = predicted_sentence


predict()
