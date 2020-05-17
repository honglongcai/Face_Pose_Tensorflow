import tensorflow.compat.v1 as tf

from data_loader import DataLoader
from network import model_fn
from cfg import FLAGS


def main(argv):
    input_fn = DataLoader(FLAGS.image_path)
    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)
    model.train(input_fn=lambda: input_fn(mode=tf.estimator.ModeKeys.TRAIN,
                                          batch_size=FLAGS.train_batch_size),
                steps=FLAGS.train_steps)
    model.evaluate(input_fn=lambda: input_fn(mode=tf.estimator.ModeKeys.EVAL,
                                             batch_size=FLAGS.eval_batch_size))


if __name__ == '__main__':
    tf.app.run(main=main)
