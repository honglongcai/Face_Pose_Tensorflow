import argparse


parser = argparse.ArgumentParser(description='parameters for training.')

parser.add_argument('--image_path', type=str,
                    default='./data/facepose_images')

parser.add_argument('--model_dir', type=str,
                    default='./model/')

parser.add_argument('--train_batch_size', type=int,
                    default=64)

parser.add_argument('--train_steps', type=int,
                    default=50000)

parser.add_argument('--eval_batch_size', type=int,
                    default=16)

parser.add_argument('--lr_values', type=float,
                    nargs='+', default=[0.001, 0.0001])

parser.add_argument('--lr_decay_at_step', type=int,
                    nargs='+', default=[20000])


FLAGS, unparsed = parser.parse_known_args()