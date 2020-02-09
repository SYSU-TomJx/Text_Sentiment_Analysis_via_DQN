import tensorflow as tf


class Network(tf.keras.Model):
    '''
    Definition of Bi-directional LSTM based model via functional API, using tf.keras.applications.
    '''

    def __init__(self, action_nums, embeddings_matrix, maxlen=100, encode_size=64, dp_rate=0.2):
        super(Network, self).__init__()
        self.actions = action_nums  # actions

        # Model struction
        self.embds = tf.keras.layers.Embedding(embeddings_matrix.shape[0],
                                               embeddings_matrix.shape[1],
                                               weights=[embeddings_matrix],
                                               input_length=maxlen,
                                               trainable=True)
        self.spatial_dp = tf.keras.layers.SpatialDropout1D(dp_rate)
        self.lstm_1 = tf.keras.layers.CuDNNLSTM(encode_size,
                                                return_sequences=True)
        self.bi_lstm_1 = tf.keras.layers.Bidirectional(self.lstm_1)
        self.max_pool = tf.keras.layers.GlobalMaxPool1D()

        self.logits = tf.keras.layers.Dense(self.actions, name='q_outputs')
        self.multipy = tf.keras.layers.Multiply()
        self.sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True), name='q_value')

    def call(self, inputs, training=None):
        # Construction
        x = inputs[0]
        act = inputs[1]

        x = self.embds(x)
        x = self.spatial_dp(x)
        x = self.bi_lstm_1(x)
        x = self.max_pool(x)
        logits = self.logits(x)
        q_val = self.multipy([act, logits])
        q_val = self.sum(q_val)

        return q_val


class LSTM(tf.keras.Model):
    '''
    Definition of Bi-directional LSTM based model via functional API, using tf.keras.applications.
    '''

    def __init__(self, class_nums, embeddings_matrix,  maxlen=100, encode_size=64, conv_filters=64, dp_rate=0.2):
        super(LSTM, self).__init__()
        self.classes = class_nums  # actions

        # Model struction
        self.embds = tf.keras.layers.Embedding(embeddings_matrix.shape[0],
                                               embeddings_matrix.shape[1],
                                               weights=[embeddings_matrix],
                                               input_length=maxlen,
                                               trainable=True)
        self.spatial_dp = tf.keras.layers.SpatialDropout1D(dp_rate)
        self.lstm_1 = tf.keras.layers.CuDNNLSTM(encode_size,
                                                return_sequences=True)
        self.bi_lstm_1 = tf.keras.layers.Bidirectional(self.lstm_1)
        self.max_pool = tf.keras.layers.GlobalMaxPool1D()

        self.logits = tf.keras.layers.Dense(self.classes, name='logits', activation='softmax')

    def call(self, inputs, training=None):
        # Construction
        x = inputs
        x = self.embds(x)
        x = self.spatial_dp(x)
        x = self.bi_lstm_1(x)
        x = self.max_pool(x)
        logits = self.logits(x)

        return logits