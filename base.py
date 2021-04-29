import tensorflow as tf
from argparse import ArgumentParser
import numpy as np
import os

from models import *

from keras.callbacks import ModelCheckpoint


### command line arguments ###
parser = ArgumentParser()
# parser.add_argument('--model', help='deep learning model to apply')

parser.add_argument('--model',
                    default='cifar10_CNN', help='deep learning model')

parser.add_argument('--gpu', type=int,
                    default='0', help='gpu to use',
                    choices=[0,1,2,3])
parser.add_argument('--seed', type=int,
                    default=0, help='random seed')
parser.add_argument('--epochs', type=int,
                    default=100, help='epochs of model train')
parser.add_argument('--batch', type=int,
                    default=64, help='batch of model train')
parser.add_argument('--optimizer',
                    default='adam', help='optimizer method')


args = parser.parse_args()

tf.random.set_seed(args.seed)
np.random.seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.gpu)    # silence some tensorflow messages
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)

# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)



cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

x_train, x_test = x_train / 255.0, x_test / 255.0


if not os.path.exists('models/ckpt'):
    os.mkdir('models/ckpt')

checkpoint_path = f'models/ckpt/{args.model}.ckpt'
model_dir = f'models/{args.model}'

cifar10_model = eval(args.model)()

if os.path.exists(model_dir):
    cifar10_model = tf.keras.models.load_model(model_dir)

else:

    checkpoint = ModelCheckpoint(checkpoint_path, 
                                save_best_only=True, 
                                save_weights_only=True, 
                                monitor='val_loss',
                                verbose=1)

    cifar10_model.model.compile(optimizer='adam',
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

    history = cifar10_model.model.fit(x_train, y_train,
                                batch_size=args.batch,
                                epochs=args.epochs,
                                validation_data=(x_test, y_test),
                                shuffle=True,
                                callbacks=[checkpoint],

    )

    cifar10_model.model.save(model_dir)

cifar10_model = eval(args.model)()

cifar10_model.model.evaluate(x_test, y_test)