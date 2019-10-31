from model import TransformerModel, load_embeddings
from prepare_data import getDataset
import tensorflow as tf
import os
import tensorflow_datasets as tfds
# Maximum sentence length
#from train import dataset_train

MAX_LENGTH = 40
BATCH_SIZE = 512
BUFFER_SIZE = 20000

"""
def evalOnTestDataset():
    tokenizer = tfds.features.text.SubwordTextEncoder()
    tokenizer.load_from_file("vocab.txt")
    [dataset_test, VOCAB_SIZE, tokenizer, _, _] = getDataset(MAX_LENGTH, BUFFER_SIZE, BATCH_SIZE,
                                                             "test", tokenizer=tokenizer)

    emb_matrix = load_embeddings(vocab_size=VOCAB_SIZE, tokenizer=tokenizer)

    Transformer = TransformerModel(max_length=MAX_LENGTH, vocab_size=VOCAB_SIZE, embedding_matrix=emb_matrix)
    model_test = Transformer.model

    checkpoint_path = "checkpoints/model_ckeckpoint.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # model = tf.keras.utils.multi_gpu_model(model, gpus=2)
    #model_test.compile(optimizer=Transformer.optimizer, loss=Transformer.loss_function, metrics=[Transformer.accuracy])

    model_test.evaluate(dataset_test, callbacks=[cp_callback], verbose=1)
"""