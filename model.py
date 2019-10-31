from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = "Lygerakis Fotios"
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import math
    import fasttext
    import tensorflow as tf
    import numpy as np
    import yaml
    import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.random.set_seed(1234)


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)

    return output


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            embedding_matrix,
            name="encoder"):
    vocab_size = embedding_matrix.shape[0]
    d_model = embedding_matrix.shape[1]
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model,
                                           weights=[embedding_matrix], trainable=False)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': look_ahead_mask
    })
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")(inputs={
        'query': attention1,
        'key': enc_outputs,
        'value': enc_outputs,
        'mask': padding_mask
    })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            embedding_matrix,
            name='decoder'):
    vocab_size = embedding_matrix.shape[0]
    d_model = embedding_matrix.shape[1]

    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model,
                                           weights=[embedding_matrix], trainable=False)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                embedding_matrix,
                name="transformer"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
        output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)
    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        embedding_matrix=embedding_matrix)(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        embedding_matrix=embedding_matrix,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


class TransformerModel:

    def __init__(self, max_length, vocab_size, embedding_matrix):
        self.MAX_LENGTH = max_length

        # Hyper-parameters
        self.VOCAB_SIZE = vocab_size
        with open('config.yaml') as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)["model"]
        self.NUM_LAYERS = int(model_config["NUM_LAYERS"])
        self.D_MODEL = int(model_config["D_MODEL"])
        self.NUM_HEADS = int(model_config["NUM_HEADS"])
        self.UNITS = int(model_config["UNITS"])
        self.DROPOUT = int(model_config["DROPOUT"])
        self.embedding_matrix = embedding_matrix
        self.initial_learning_rate = float(model_config["lr"]["initial_LR"])
        self.decay_steps = int(model_config["lr"]["decay_steps"])
        self.decay_rate = float(model_config["lr"]["decay_rate"])
        self.staircase = bool(model_config["lr"]["staircase"])
        self.initial_learning_rate = float(model_config["lr"]["initial_LR"])

        tf.keras.backend.clear_session()

        self.model = transformer(
            vocab_size=self.VOCAB_SIZE,
            num_layers=self.NUM_LAYERS,
            units=self.UNITS,
            d_model=self.D_MODEL,
            num_heads=self.NUM_HEADS,
            dropout=self.DROPOUT,
            embedding_matrix=self.embedding_matrix
        )

        # self.learning_rate = CustomSchedule(self.D_MODEL)
        # self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        #    self.initial_learning_rate,
        #    decay_steps=self.decay_steps,
        #    decay_rate=self.decay_rate,
        #    staircase=self.staircase)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate,
                                                  beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def loss_function(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, self.MAX_LENGTH - 1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    def accuracy(self, y_true, y_pred):
        # ensure labels have shape (batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, self.MAX_LENGTH - 1))
        accuracy = tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)
        return accuracy


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model=300, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    # @tf.function
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def load_embeddings(vocab_size, tokenizer, language):
    embeddings_index = {}
    fasttext_model = None
    if language.lower() == "greek":
        fasttext_model = fasttext.load_model("embeddings/cc.el.300.bin")
    elif language.lower() == "english":
        fasttext_model = fasttext.load_model("embeddings/cc.en.300.bin")
    else:
        print("Language: " + language + "is not supported")
        print("Try \"Greek\" or \"English\".")

    with open('config.yaml') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)["model"]
    EMBEDDING_DIM = int(model_config["D_MODEL"])
    if fasttext_model is not None:
        for word in tokenizer.subwords:
            embeddings_index[word] = fasttext_model[word]
        EMBEDDING_DIM = len(embeddings_index["and_"])

    embedding_matrix = np.zeros((vocab_size + 1, EMBEDDING_DIM))
    for word in list(embeddings_index.keys()):
        i = tokenizer.encode(word)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def exp_decay_scedule(epoch, lr):
    with open('config.yaml') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)["model"]
    decay_steps = int(model_config["lr"]["decay_steps"])
    decay_rate = float(model_config["lr"]["decay_rate"])
    staircase = bool(model_config["lr"]["staircase"])

    if staircase:
        learning_rate = lr * math.pow(decay_rate, round(float(epoch) / float(decay_steps)))
    else:
        learning_rate = lr * math.pow(decay_rate, (float(epoch) / float(decay_steps)))
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)

    return learning_rate


def custom_scedule(epoch):
    with open('config.yaml') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)["model"]
    warmup_steps = int(model_config["lr"]["warmup_steps"])
    if epoch != 0:
        arg1 = 1 / math.sqrt(epoch)
        arg2 = epoch * (warmup_steps ** -1.5)
        learning_rate = 1 / math.sqrt(300) * min(arg1, arg2)
    else:
        learning_rate = float(model_config["lr"]["initial_LR"])

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)

    return learning_rate
