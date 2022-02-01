import tensorflow as tf
import tf_transformer_models
import utils
import datetime
from tensorflow import keras


class Tied_Embedding(keras.layers.Layer):
    def __init__(self, vocab_size, embedding_depth):
        super(Tied_Embedding, self).__init__()
        self._embedding = keras.layers.Embedding(vocab_size, embedding_depth)

    def call(self, inputs):
        return self._embedding(inputs)

    def output_embedding(self, inputs):
        return tf.matmul(inputs, self._embedding.embeddings, transpose_b=True)


def create_lstm_input_model(
    desc, batch_size, embedding_depth, lstm_layers, lstm_depth, num_inputs
):
    # The LSTM input model is called with:
    #     - the embedding of the timestep input token (i.e. the output of the last step) of shape (batch_size, embedding_depth)
    #     - the stack of features with size (batch_size, num_distractors+1, feature_embedding_size)
    #     - the last hidden state h_(t-1) of the LSTM block with shape (batch_size, lstm_layers, lstm_depth)
    ########################################################################################################
    if desc == "embedding_only":
        embedding_func = lambda embedding_features_hidden: embedding_features_hidden[0]
        return embedding_func
    if desc == "simple_attention":
        # basically scaled cross product attention with only one element in query
        # weird construction because hidden states are of shape (lstm_layers, batch_size, lstm_depth)
        hidden = keras.Input(shape=(batch_size, lstm_depth), batch_size=lstm_layers)
        embedding = keras.Input(shape=(embedding_depth,), batch_size=batch_size)
        features = keras.Input(shape=(num_inputs, 64), batch_size=batch_size)

        # using lstm depth as depth of the attention mapping
        hidden_query = keras.layers.Dense(lstm_depth)(
            tf.concat((tf.unstack(hidden, axis=0)), axis=-1)
        )
        features_key = keras.layers.Dense(lstm_depth)(features)

        hidden_query = tf.expand_dims(hidden_query, axis=1)
        # feature_keys of shape (b_s, num_inputs, lstm_depth)
        pre_scores = tf.reduce_sum(features_key * hidden_query, axis=-1) / tf.sqrt(
            tf.cast(lstm_depth, dtype=tf.float32)
        )
        att_scores = tf.expand_dims(tf.nn.softmax(pre_scores), axis=-1)
        # key=value
        attended = tf.reduce_sum(att_scores * features_key, axis=1)

        return keras.Model(inputs=[embedding, features, hidden], outputs=attended)


def create_lstm_output_model(desc, size):
    if desc == "dense":
        return keras.layers.Dense(units=size, activation="tanh")


def create_encoder(desc):
    if desc == "two_layer_transformer_model":
        encoder = tf_transformer_models.Encoder(
            num_layers=2, d_model=64, num_heads=4, dff=64
        )
        return encoder


def create_decoder(desc):
    if desc == "two_layer_transformer_model":
        decoder = tf_transformer_models.Decoder(
            num_layers=2, d_model=64, num_heads=4, dff=64
        )
        return decoder


def lstm_attention_receiver(
    num_inputs=8,
    feature_depth=2048,
    max_seq_len=45,
    vocab_size=2000,
    embedding_depth=256,
    lstm_depth=256,
    lstm_layers=2,
    end_token_idx=4,
    batch_size=32,
    decoder_name="two_layer_transformer_model",
    lstm_input_model_name="simple_attention",
    lstm_output_model_name="dense",
    name="attention_lstm_speaker",
    name_is_suffix=False,
):

    # features has all distractors plus the target in a tensor of shape (batch_size, num_distractors+1, feature_depth+1)
    # num_distractors+1 because it also includes the target, feature_depth+1 because a dimension encoding the target/distractor property is added
    features = keras.Input(shape=(num_inputs, feature_depth))
    token_sequence = keras.Input(shape=(max_seq_len,), dtype=tf.int64)

    message_embedding = keras.layers.Embedding(vocab_size, embedding_depth)(
        token_sequence
    )

    lstms = [
        tf.keras.layers.LSTM(units=lstm_depth, return_sequences=True)
        for _ in range(lstm_layers)
    ]

    for i in range(lstm_layers):
        message_embedding = lstms[i](message_embedding)

    # scan messages for end of sequence tokens
    eos_mask = utils.eos_mask(end_token_idx, token_sequence, batch_size)
    # notice that there could be multiple eos tokens per sequence, which is why it has to be done this more complicated way
    eos_idxs = tf.reduce_sum(tf.cast(eos_mask, dtype=tf.int64), axis=-1)

    # should now return with shape (batch_size, embedding_depth)
    eos_encodings = tf.gather(message_embedding, eos_idxs, batch_dims=1)

    decoder = create_decoder(decoder_name)
    decoder_outs = decoder(features, eos_encodings)

    predictions = keras.layers.Dense(1, activation="linear")(decoder_outs)
    logits = tf.squeeze(predictions)

    receiver_model = keras.Model(inputs=(features, token_sequence), outputs=logits)
    return receiver_model


def attention_lstm_speaker(
    num_inputs=8,
    feature_depth=2048,
    max_seq_len=45,
    vocab_size=2000,
    embedding_depth=256,
    lstm_depth=256,
    lstm_layers=2,
    start_token_idx=3,
    batch_size=32,
    target_embedding_size=64,
    encoder_name="two_layer_transformer_model",
    lstm_input_model_name="simple_attention",
    lstm_output_model_name="dense",
    name="attention_lstm_speaker",
    name_is_suffix=False,
):
    if name_is_suffix:
        name = "attention_lstm_speaker_" + name

    class Step_wise_lstm_sampling_output(keras.Model):
        def __init__(
            self,
            max_seq_len,
            vocab_size,
            batch_size,
            lstm_layers,
            lstm_depth,
            embedding,
            lstms,
            lstm_input_model,
            lstm_output_model,
        ):
            super(Step_wise_lstm_sampling_output, self).__init__()
            self.vocab_size = vocab_size
            self.max_seq_len = max_seq_len
            self.embedding = embedding
            self.batch_size = batch_size
            self.lstms = lstms
            self.lstm_layers = lstm_layers
            self.lstm_depth = lstm_depth
            self.lstm_input_model = lstm_input_model
            self.lstm_output_model = lstm_output_model

        def call(
            self,
            starting_token,
            initial_state_c,
            initial_state_h,
            encoding,
        ):

            logits_agg = tf.TensorArray(
                dtype=tf.float32, size=max_seq_len, element_shape=(None, vocab_size)
            )
            token_agg = tf.TensorArray(
                dtype=tf.int64, size=max_seq_len, element_shape=(None, 1)
            )

            c_agg = tf.TensorArray(
                dtype=tf.float32,
                size=lstm_layers,
                element_shape=(batch_size, lstm_depth),
            )
            h_agg = tf.TensorArray(
                dtype=tf.float32,
                size=self.lstm_layers,
                element_shape=(batch_size, lstm_depth),
            )

            c_all = tf.stack(initial_state_c)
            h_all = tf.stack(initial_state_h)

            # set the current token to
            current_token = starting_token
            for i in tf.range(max_seq_len):
                token_embedding = tf.squeeze(self.embedding(current_token))
                lstm_input = self.lstm_input_model((token_embedding, encoding, h_all))

                for j in range(self.lstm_layers):
                    lstm_input, states = self.lstms[j](lstm_input, [h_all[j], c_all[j]])
                    h_agg = h_agg.write(j, states[0])
                    c_agg = c_agg.write(j, states[1])

                h_all = h_agg.stack()
                c_all = c_agg.stack()
                lstm_out = self.lstm_output_model(lstm_input)
                logits = self.embedding.output_embedding(lstm_out)

                current_token = tf.random.categorical(
                    logits,
                    num_samples=1,
                    dtype=tf.int64,
                )

                logits_agg = logits_agg.write(i, logits)
                token_agg = token_agg.write(i, current_token)

            logits_out = (
                logits_agg.stack()
            )  # shape (sequence_length, batch_size, vocab_size)
            logits_out = tf.transpose(logits_out, perm=[1, 0, 2])
            output_sequence = token_agg.stack()  # size(sequence_length, batch_size,1)
            output_sequence = tf.transpose(
                tf.squeeze(output_sequence)
            )  # perm is implicitly [1,0]

            return logits_out, output_sequence

    class Step_wise_lstm_given_output(keras.Model):
        def __init__(
            self,
            max_seq_len,
            vocab_size,
            batch_size,
            lstm_layers,
            lstm_depth,
            embedding,
            lstms,
            lstm_input_model,
            lstm_output_model,
        ):
            super(Step_wise_lstm_given_output, self).__init__()
            self.vocab_size = vocab_size
            self.max_seq_len = max_seq_len
            self.embedding = embedding
            self.batch_size = batch_size
            self.lstms = lstms
            self.lstm_layers = lstm_layers
            self.lstm_depth = lstm_depth
            self.lstm_input_model = lstm_input_model
            self.lstm_output_model = lstm_output_model

        def call(self, token_sequence, initial_state_c, initial_state_h, encoding):

            logits_agg = tf.TensorArray(
                dtype=tf.float32, size=max_seq_len, element_shape=(None, vocab_size)
            )

            c_agg = tf.TensorArray(
                dtype=tf.float32,
                size=lstm_layers,
                element_shape=(batch_size, lstm_depth),
            )
            h_agg = tf.TensorArray(
                dtype=tf.float32,
                size=self.lstm_layers,
                element_shape=(batch_size, lstm_depth),
            )

            c_all = tf.stack(initial_state_c)
            h_all = tf.stack(initial_state_h)

            for i in tf.range(max_seq_len):
                current_token = tf.expand_dims(token_sequence[:, i], axis=-1)
                token_embedding = tf.squeeze(self.embedding(current_token))
                tf.squeeze(token_embedding)

                lstm_input = self.lstm_input_model((token_embedding, encoding, h_all))

                for j in range(self.lstm_layers):
                    lstm_input, states = self.lstms[j](lstm_input, [h_all[j], c_all[j]])
                    h_agg = h_agg.write(j, states[0])
                    c_agg = c_agg.write(j, states[1])

                h_all = h_agg.stack()
                c_all = c_agg.stack()
                lstm_out = self.lstm_output_model(lstm_input)
                logits = self.embedding.output_embedding(lstm_out)

                logits_agg = logits_agg.write(i, logits)

            logits_out = (
                logits_agg.stack()
            )  # shape (sequence_length, batch_size, vocab_size)
            logits_out = tf.transpose(logits_out, perm=[1, 0, 2])

            return logits_out

    with tf.name_scope(name):

        # features has all distractors plus the target in a tensor of shape (batch_size, num_distractors+1, feature_depth+1)
        # num_distractors+1 because it also includes the target, feature_depth+1 because a dimension encoding the target/distractor property is added
        features = tf.keras.Input(shape=(num_inputs, feature_depth + 1))
        target = tf.keras.Input(shape=(feature_depth))  #
        token_sequence = keras.Input(shape=(max_seq_len), dtype=tf.int64)

        embedding = Tied_Embedding(vocab_size, embedding_depth)
        encoder = create_encoder(encoder_name)  # ToDo
        encoding = encoder(features)

        # create initial states encoding the target for the lstm

        target_embedding = tf.keras.layers.Dense(
            units=target_embedding_size, activation="tanh"
        )(target)
        initial_state_c = [
            keras.layers.Dense(units=lstm_depth, activation="tanh")(target_embedding)
            for _ in range(lstm_layers)
        ]
        initial_state_h = [
            keras.layers.Dense(units=lstm_depth, activation="tanh")(target_embedding)
            for _ in range(lstm_layers)
        ]
        # create lstm stack
        lstms = [keras.layers.LSTMCell(units=lstm_depth) for _ in range(lstm_layers)]

        lstm_input_model = create_lstm_input_model(
            lstm_input_model_name,
            batch_size,
            embedding_depth,
            lstm_layers,
            lstm_depth,
            num_inputs,
        )

        lstm_output_model = create_lstm_output_model(
            lstm_output_model_name, size=embedding_depth
        )

        starting_token = (
            tf.zeros(shape=(batch_size, 1), dtype=tf.int64) + start_token_idx
        )

        so_unroll = Step_wise_lstm_sampling_output(
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            batch_size=batch_size,
            lstm_layers=lstm_layers,
            lstm_depth=lstm_depth,
            embedding=embedding,
            lstms=lstms,
            lstm_input_model=lstm_input_model,
            lstm_output_model=lstm_output_model,
        )

        go_unroll = Step_wise_lstm_given_output(
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            batch_size=batch_size,
            lstm_layers=lstm_layers,
            lstm_depth=lstm_depth,
            embedding=embedding,
            lstms=lstms,
            lstm_input_model=lstm_input_model,
            lstm_output_model=lstm_output_model,
        )

        so_logits, so_seq = so_unroll(
            starting_token,
            initial_state_c,
            initial_state_h,
            encoding,
        )
        go_logits = go_unroll(
            token_sequence, initial_state_c, initial_state_h, encoding
        )

        so_model = keras.Model(inputs=[features, target], outputs=[so_logits, so_seq])
        go_model = keras.Model(
            inputs=[features, target, token_sequence], outputs=[go_logits]
        )
        return so_model, go_model


if __name__ == "__main__":

    receiver_model = lstm_attention_receiver()
    self_sampling_model, given_sequence_model = attention_lstm_speaker()

    @tf.function
    def test():
        features = tf.zeros(shape=(32, 8, 2049))
        target = tf.zeros(shape=(32, 2048))
        token_sequence = tf.zeros(shape=(32, 20), dtype=tf.int64)
        self_sampling_model((features, target))
        given_sequence_model((features, target, token_sequence))
        features = tf.zeros(shape=(32, 8, 2048))

        receiver_model((features, token_sequence))

    tf.summary.trace_on(graph=True)
    test()
    print(receiver_model.summary(expand_nested=True))
    print(self_sampling_model.summary(expand_nested=True))
    print(given_sequence_model.summary(expand_nested=True))

    for m in [given_sequence_model, self_sampling_model]:
        for l in m.layers:
            for var in l.trainable_variables:
                print(var.name, var.shape)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer("test_logs/" + current_time)
    with writer.as_default():
        tf.summary.trace_export(
            name="test_trace", step=0, profiler_outdir="test_logs/" + current_time
        )
