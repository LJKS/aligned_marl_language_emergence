from data import *
import tensorflow as tf
from tensorflow import keras
import datetime


def feat_to_seq_model(vocab_size=2000, units=256, layers=2):
    shifted_sequence = keras.Input(shape=(None,), dtype=tf.float32)
    features = keras.Input(shape=2048)
    features_h = keras.layers.Dense(units=units, activation="tanh")(features)
    features_c = keras.layers.Dense(units=units, activation="tanh")(features)
    embedding_layer = keras.layers.Embedding(vocab_size, units)
    x = embedding_layer(shifted_sequence)
    initial_state = [features_h, features_c]
    for layer in range(layers):
        lstm_layer = keras.layers.LSTM(units=units, return_sequences=True)
        # x = lstm_layer(inputs=x, initial_state=[features_h, features_c])
        x = lstm_layer(inputs=x, initial_state=initial_state)
    x = tf.matmul(x, embedding_layer.embeddings, transpose_b=True)
    x = tf.nn.softmax(x)

    model = keras.Model(inputs=[features, shifted_sequence], outputs=x)
    return model


def train_feat_to_seq_model(
    batch_size=16, epochs=20, learning_rate=0.001, logs_per_epoch=10
):
    model = feat_to_seq_model()
    train_ds, test_ds = feature_capAsIdxs_id_ds()

    def prepare_ds(ds):
        def empty_start_token_and_encode(seq):
            seq = tf.concat([tf.ones(shape=(1,), dtype=tf.int64), seq], axis=0)
            return seq

        ds = ds.map(
            lambda feature, caption, id: (
                feature,
                empty_start_token_and_encode(caption),
                tf.ones_like(caption, dtype=tf.float32),
            )
        )
        ds = ds.shuffle(20000).padded_batch(batch_size).prefetch(20)
        return ds

    train_ds = prepare_ds(train_ds)
    test_ds = prepare_ds(test_ds)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.SparseCategoricalCrossentropy(reduction="none")

    train_loss = tf.keras.metrics.Mean("loss", dtype=tf.float32)
    train_accuracy = tf.keras.metrics.Mean("perplexity", dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean("loss", dtype=tf.float32)
    test_accuracy = tf.keras.metrics.Mean("perplexity", dtype=tf.float32)

    @tf.function
    def train_step(feature, sequence, mask):
        with tf.GradientTape() as tape:
            x = [feature, sequence[:, :-1]]
            prediction = model(x)
            cce = loss(sequence[:, 1:], prediction) * mask
            l = tf.reduce_sum(cce) / tf.reduce_sum(mask)
        grads = tape.gradient(l, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(l)
        train_accuracy(tf.exp(l))

    @tf.function
    def test_step(feature, sequence, mask):
        x = [feature, sequence[:, :-1]]
        prediction = model(x)
        cce = loss(sequence[:, 1:], prediction) * mask
        l = tf.reduce_sum(cce) / tf.reduce_sum(mask)

        test_loss(l)
        test_accuracy(tf.exp(l))

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "logs/feat_to_seq/" + current_time + "/train"
    test_log_dir = "logs/feat_to_seq/" + current_time + "/test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # for ds wiht unknown/undefined cardinality
    def get_ds_cardinality(ds):
        card = 0
        for i, _ in enumerate(ds):
            card = i
        return card

    # log 10 times per epoch
    train_log_steps = round(get_ds_cardinality(train_ds) / logs_per_epoch)
    test_log_steps = round(get_ds_cardinality(test_ds) / logs_per_epoch)

    for epoch in range(epochs):
        # train

        step = 0
        for i, elem in enumerate(train_ds):
            feature, sequence, mask = elem
            train_step(feature, sequence, mask)
            # print(i % train_log_steps, i, train_log_steps, train_ds.cardinality())
            if (i % train_log_steps) == 0:
                with train_summary_writer.as_default():
                    print(
                        f"Epoch {epoch} || Step {step} || CCE {train_loss.result().numpy():.3f} || PP {train_accuracy.result().numpy() :.3f} (TRAIN)"
                    )
                    tf.summary.scalar(
                        "cce loss",
                        train_loss.result(),
                        step=epoch * logs_per_epoch + step,
                    )
                    tf.summary.scalar(
                        "perplexity",
                        train_accuracy.result(),
                        step=epoch * logs_per_epoch + step,
                    )

                    step = step + 1

                    train_loss.reset_states()
                    train_accuracy.reset_states()

        # test
        for i, elem in enumerate(test_ds):
            feature, sequence, mask = elem
            test_step(feature, sequence, mask)
            # print(i % test_log_steps)
            # if (i % test_log_steps) == 0:
        with test_summary_writer.as_default():
            print(
                f"Epoch {epoch} || Step {step} || CCE {test_loss.result().numpy():.3f} || PP {test_accuracy.result().numpy() :.3f} (TEST)"
            )
            tf.summary.scalar(
                "cce loss", test_loss.result(), step=epoch * logs_per_epoch
            )
            tf.summary.scalar(
                "perplexity", test_accuracy.result(), step=epoch * logs_per_epoch
            )

            step = step + 1

            test_loss.reset_states()
            test_accuracy.reset_states()


if __name__ == "__main__":
    train_feat_to_seq_model()
