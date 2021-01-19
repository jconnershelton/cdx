import os
import inout
import random
import platform
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_SPLIT = None
EPOCHS = None
REQUIRED_ACCURACY = None
REQUIRED_LOSS = None
CONFIG_FILE = None


def draw_model_summary(layers):
    print(inout.TerminalCodes.CLEAR, end='')
    print(f'{inout.TerminalCodes.BOLD}{inout.TerminalCodes.UNDERLINE}Layers{inout.TerminalCodes.END}')
    for layer in layers: print(layer)

    print()


def get_frozen_graph(model):
    func = tf.function(lambda x: model(x))
    func = func.get_concrete_function(x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    frozen = convert_variables_to_constants_v2(func)
    frozen.graph.as_graph_def()

    return frozen.graph


def shuffle_split_data(data, labels, train_split):
    seed = random.randint(0, 256)
    data = np.expand_dims(tf.random.shuffle(tf.convert_to_tensor(data), seed=seed), axis=-1)
    labels = tf.random.shuffle(tf.convert_to_tensor(labels), seed=seed)

    split_idx = round(len(data) * train_split)
    return data[:split_idx], labels[:split_idx], data[split_idx:], labels[split_idx:]


def train():
    images, mappings, labels = inout.from_cdx_file(inout.INPUT if inout.INPUT else inout.get_input('Path to CDX: '))
    images = [image / 255.0 for image in images]

    try: train_split = float(TRAIN_SPLIT if TRAIN_SPLIT else inout.get_input('Train split [float 0-1]: '))
    except ValueError: inout.err('Invalid train split. Must be float.')
    if not 0 <= train_split <= 1: inout.err('Invalid train split. Must be between 0 and 1 inclusive.')

    try: epochs = int(EPOCHS if EPOCHS else inout.get_input('Epochs [integer > 0]: '))
    except ValueError: inout.err('Invalid epoch count. Must be integer.')
    if epochs < 1: inout.err('Invalid epoch count. Must be positive integer.')

    try: required_accuracy = float(REQUIRED_ACCURACY if REQUIRED_ACCURACY else 0)
    except ValueError: inout.err('Invalid required accuracy. Must be float.')

    try: required_loss = float(REQUIRED_LOSS if REQUIRED_LOSS else 'inf')
    except ValueError: inout.err('Invalid required loss. Must be float.')

    if CONFIG_FILE:
        config_file = open(CONFIG_FILE, 'r')
        layers = [line[:-1] for line in config_file if line[:-1]]
    else:
        layers = []
        while True:
            draw_model_summary(layers)

            layer = inout.get_input(f'New layer [Keras layer expression]{" (blank if done)" if len(layers) > 0 else ""}: ')
            if not layer: break

            if len(layers) == 0: layer = layer[:-1] + f'{", " if "," in layer else ""}input_shape={images[0].shape + (1,)})'
            layers.append(layer)

    if len(layers) == 0: inout.err('Must have at least one layer.')

    layers.append(f"Dense({len(mappings)}, activation='{inout.get_input('Output layer activation: ')}')")
    draw_model_summary(layers)

    while True:
        tf.keras.backend.clear_session()
        train_data, train_labels, test_data, test_labels = shuffle_split_data(images, labels, train_split)

        model = tf.keras.models.Sequential()
        for layer in layers: model.add(eval(f'tf.keras.layers.{layer}'))
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        print()
        model.fit(train_data, train_labels, epochs=epochs)
        loss, accuracy = model.evaluate(test_data, test_labels, verbose=False)

        print()
        print(f'Testing Accuracy: {accuracy}')
        print(f'Testing Loss: {loss}')

        if accuracy >= required_accuracy and loss <= required_loss: break

    save_path = inout.OUTPUT if inout.OUTPUT else inout.get_input('Path to save model (leave blank to discard): ')
    if not save_path: return

    extension = save_path.split('.')[-1]

    if extension == 'pb':
        if platform.system() == 'Windows': separator = '\\'
        else: separator = '/'

        save_path = save_path.split(separator)
        save_dir = separator.join(save_path[:-1]) if save_path[:-1] else '.'
        save_file = save_path[-1]

        tf.io.write_graph(graph_or_graph_def=get_frozen_graph(model), logdir=save_dir, name=save_file, as_text=False)
    else: model.save(save_path)