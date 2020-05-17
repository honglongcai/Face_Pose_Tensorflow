import tensorflow.compat.v1 as tf

from cfg import FLAGS


def network(features, mode):
    features = tf.layers.conv2d(inputs=features, kernel_size=1, filters=16, strides=(1, 1),
                                padding='same', kernel_initializer=tf.variance_scaling_initializer())
    features = tf.nn.relu(features)
    features = tf.layers.conv2d(inputs=features, kernel_size=3, filters=16, strides=(2, 2),
                                padding='same', kernel_initializer=tf.variance_scaling_initializer())
    features = tf.nn.relu(features)
    features = tf.layers.conv2d(inputs=features, kernel_size=1,
                                filters=32, strides=(1, 1), padding='same',
                                kernel_initializer=tf.variance_scaling_initializer())
    features = tf.nn.relu(features)
    features = tf.nn.max_pool2d(features, ksize=2, strides=2, padding='SAME')
    features = tf.reshape(features, [-1, 10 * 10 * 32])
    features = tf.layers.dense(
        inputs=features,
        units=512,
        kernel_initializer=tf.variance_scaling_initializer())
    features = tf.nn.relu(features)
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    features = tf.layers.dropout(features, rate=0.2, training=training)
    logits = tf.layers.dense(
        inputs=features,
        units=9,
        kernel_initializer=tf.variance_scaling_initializer())
    return logits


# model_fn
def model_fn(features, labels, mode):
    logits = network(features, mode=mode)
    predictions = tf.argmax(logits, axis=1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.piecewise_constant(global_step, boundaries=FLAGS.lr_decay_at_step, values=FLAGS.lr_values)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(
        loss, global_step=tf.train.get_global_step())
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={'accuracy': accuracy})

