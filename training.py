import tensorflow as tf
import agents
import data
import utils
import datetime


def train_sender_model_supervised(
    batch_size=48, num_inputs=8, max_seq_len=45, num_epochs=40
):
    _, model = agents.attention_lstm_speaker(
        batch_size=batch_size, num_inputs=num_inputs, max_seq_len=max_seq_len
    )
    ds_train, ds_test = data.feature_capAsIdxs_id_ds()

    def prep_data(ds, eos_token, num_inputs, batch_size):
        ds = ds.filter(lambda feature, idxs, id: tf.shape(idxs)[0] <= max_seq_len)
        ds = ds.shuffle(100000).padded_batch(
            num_inputs, drop_remainder=True, padded_shapes=([2048], [max_seq_len], [])
        )
        # maybe keep id and filter rare cases of showing the same image also as a distractor out?
        ds = ds.filter(
            lambda feature, idxs, id: tf.reduce_sum(tf.unique(id)[0])
            == tf.reduce_sum(id)
        )

        ds = ds.map(
            lambda feature, idxs, id: (
                feature,
                idxs,
                utils.eos_mask(
                    eos_token,
                    idxs,
                    num_inputs,
                    first_eos_included=tf.ones((1,), dtype=tf.bool),
                ),
                id,
            )
        )
        # treat first pseudeo_batch_element as target
        # mappigng to (org_features, target_encoded_features, target_sequence, distractor_sequences, target_only_feature, target_onehot, target_mask, mask, id)
        ds = ds.map(
            lambda feature, idxs, mask, id: (
                feature,
                tf.concat(
                    [
                        tf.transpose(
                            tf.one_hot(tf.zeros(shape=(1,), dtype=tf.int32), num_inputs)
                        ),
                        feature,
                    ],
                    axis=-1,
                ),
                idxs[0, :],
                idxs[1:, :],
                feature[0, :],
                tf.squeeze(
                    tf.one_hot(tf.zeros(shape=(1,), dtype=tf.int32), depth=num_inputs)
                ),
                mask[0, :],
                mask,
                id,
            )
        )
        ds = ds.padded_batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(20)

        return ds

    ds_train = prep_data(
        ds_train, eos_token=4, num_inputs=num_inputs, batch_size=batch_size
    )
    ds_test = prep_data(
        ds_test, eos_token=4, num_inputs=num_inputs, batch_size=batch_size
    )

    # Utils for logging the training
    train_loss = tf.keras.metrics.Mean("loss", dtype=tf.float32)
    train_accuracy = tf.keras.metrics.Mean("perplexity", dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean("loss", dtype=tf.float32)
    test_accuracy = tf.keras.metrics.Mean("perplexity", dtype=tf.float32)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "logs/supervised_sender/" + current_time + "/train"
    test_log_dir = "logs/supervised_sender/" + current_time + "/test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(enc_features, target_feature, target_sequence, mask):
        # target is right shifted input
        model_target_sequence = target_sequence[:, 1:]
        mask = tf.cast(mask[:, 1:], dtype=tf.float32)
        with tf.GradientTape() as tape:
            logits = model((enc_features, target_feature, target_sequence))
            logits = logits[:, :-1, :]
            l = cce_loss(model_target_sequence, logits)
            loss = l * mask
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        pp = tf.exp(tf.reduce_sum(l * mask) * (1 / tf.reduce_sum(mask)))
        train_loss(loss)
        train_accuracy(pp)

    @tf.function
    def test_step(enc_features, target_feature, target_sequence, mask):
        # target is right shifted input
        model_target_sequence = target_sequence[:, 1:]
        mask = tf.cast(mask[:, 1:], dtype=tf.float32)

        logits = model((enc_features, target_feature, target_sequence))
        logits = logits[:, :-1, :]
        l = cce_loss(model_target_sequence, logits)
        loss = l * mask
        loss = tf.reduce_mean(loss)

        pp = tf.exp(tf.reduce_sum(l * mask) * (1 / tf.reduce_sum(mask)))
        test_loss(loss)
        test_accuracy(pp)

    for epoch in range(num_epochs):
        # train
        for elem in ds_train:
            # print("training!")
            (
                org_features,
                target_encoded_features,
                target_sequence,
                distractor_sequences,
                target_only_feature,
                target_onehot,
                target_mask,
                mask,
                id,
            ) = elem
            train_step(
                target_encoded_features,
                target_only_feature,
                target_sequence,
                target_mask,
            )

        with train_summary_writer.as_default():
            print(
                f"Epoch {epoch:3.0f} || CCE {train_loss.result().numpy():.3f} || PP {train_accuracy.result().numpy() :.3f} (TRAIN)"
            )
            tf.summary.scalar(
                "cce loss",
                train_loss.result(),
                step=epoch,
            )
            tf.summary.scalar(
                "perplexity",
                train_accuracy.result(),
                step=epoch,
            )

        train_loss.reset_states()
        train_accuracy.reset_states()

        for elem in ds_test:
            (
                org_features,
                target_encoded_features,
                target_sequence,
                distractor_sequences,
                target_only_feature,
                target_onehot,
                target_mask,
                mask,
                id,
            ) = elem
            test_step(
                target_encoded_features,
                target_only_feature,
                target_sequence,
                target_mask,
            )
        with test_summary_writer.as_default():
            print(
                f"Epoch {epoch} || CCE {test_loss.result().numpy():.3f} || PP {test_accuracy.result().numpy() :.3f} (TEST)"
            )
            tf.summary.scalar("cce loss", test_loss.result(), step=epoch)
            tf.summary.scalar("perplexity", test_accuracy.result(), step=epoch)

            test_loss.reset_states()
            test_accuracy.reset_states()


def train_receiver_model_supervised():
    model = agents.lstm_attention_receiver()
    ds_train, ds_test = data.feature_capAsIdxs_id_ds()

    def prep_data(ds, eos_token, num_inputs, batch_size):
        ds = ds.filter(lambda feature, idxs, id: tf.shape(idxs)[0] <= max_seq_len)
        ds = ds.shuffle(100000).padded_batch(
            num_inputs, drop_remainder=True, padded_shapes=([2048], [max_seq_len], [])
        )
        # maybe keep id and filter rare cases of showing the same image also as a distractor out?
        ds = ds.filter(
            lambda feature, idxs, id: tf.reduce_sum(tf.unique(id)[0])
            == tf.reduce_sum(id)
        )

        ds = ds.map(
            lambda feature, idxs, id: (
                feature,
                idxs,
                utils.eos_mask(
                    eos_token,
                    idxs,
                    num_inputs,
                    first_eos_included=tf.ones((1,), dtype=tf.bool),
                ),
                id,
            )
        )
        # treat first pseudeo_batch_element as target
        # mappigng to (org_features, target_encoded_features, target_sequence, distractor_sequences, target_only_feature, target_onehot, target_mask, mask, id)
        ds = ds.map(
            lambda feature, idxs, mask, id: (
                feature,
                tf.concat(
                    [
                        tf.transpose(
                            tf.one_hot(tf.zeros(shape=(1,), dtype=tf.int32), num_inputs)
                        ),
                        feature,
                    ],
                    axis=-1,
                ),
                idxs[0, :],
                idxs[1:, :],
                feature[0, :],
                tf.squeeze(
                    tf.one_hot(tf.zeros(shape=(1,), dtype=tf.int32), depth=num_inputs)
                ),
                mask[0, :],
                mask,
                id,
            )
        )
        ds = ds.padded_batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(20)

        return ds

    ds_train = prep_data(
        ds_train, eos_token=4, num_inputs=num_inputs, batch_size=batch_size
    )
    ds_test = prep_data(
        ds_test, eos_token=4, num_inputs=num_inputs, batch_size=batch_size
    )

    # Utils for logging the training
    train_loss = tf.keras.metrics.Mean("loss", dtype=tf.float32)
    train_accuracy = tf.keras.metrics.Mean("perplexity", dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean("loss", dtype=tf.float32)
    test_accuracy = tf.keras.metrics.Mean("perplexity", dtype=tf.float32)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "logs/supervised_sender/" + current_time + "/train"
    test_log_dir = "logs/supervised_sender/" + current_time + "/test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(enc_features, input_sequence, mask):
        # target is right shifted input
        model_target_sequence = target_sequence[:, 1:]
        mask = tf.cast(mask[:, 1:], dtype=tf.float32)
        with tf.GradientTape() as tape:
            logits = model((enc_features, target_feature, target_sequence))
            logits = logits[:, :-1, :]
            l = cce_loss(model_target_sequence, logits)
            loss = l * mask
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        pp = tf.exp(tf.reduce_sum(l * mask) * (1 / tf.reduce_sum(mask)))
        train_loss(loss)
        train_accuracy(pp)

    @tf.function
    def test_step(enc_features, target_feature, target_sequence, mask):
        # target is right shifted input
        model_target_sequence = target_sequence[:, 1:]
        mask = tf.cast(mask[:, 1:], dtype=tf.float32)

        logits = model((enc_features, target_feature, target_sequence))
        logits = logits[:, :-1, :]
        l = cce_loss(model_target_sequence, logits)
        loss = l * mask
        loss = tf.reduce_mean(loss)

        pp = tf.exp(tf.reduce_sum(l * mask) * (1 / tf.reduce_sum(mask)))
        test_loss(loss)
        test_accuracy(pp)


if __name__ == "__main__":
    train_sender_model_supervised()
