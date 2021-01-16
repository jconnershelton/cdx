import os
import copy
import inout
import config
import random
import platform
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def draw_model_summary(layers, units_list, activations):
    total_layer_space = max(9, (max(len(name) for name in layers) + 3) if layers else 0)
    total_units_space = max(8, (max(len(str(units)) for units in units_list) + 3) if units_list else 0)

    print(inout.TerminalCodes.CLEAR, end='')
    print(f'{inout.TerminalCodes.BOLD}{inout.TerminalCodes.UNDERLINE}Layers{inout.TerminalCodes.END}' + ' ' * (total_layer_space - 6) + f'{inout.TerminalCodes.BOLD}{inout.TerminalCodes.UNDERLINE}Units{inout.TerminalCodes.END}' + ' ' * (total_units_space - 5) + f'{inout.TerminalCodes.BOLD}{inout.TerminalCodes.UNDERLINE}Activation{inout.TerminalCodes.END}')

    for line in range(len(layers)):
        layer = layers[line]
        units = str(units_list[line])
        activation = activations[line]

        print(layer + ' ' * (total_layer_space - len(layer)) + units + ' ' * (total_units_space - len(units)) + activation)

    print()

def get_frozen_graph(model):
    func = tf.function(lambda x: model(x))
    func = func.get_concrete_function(x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    frozen = convert_variables_to_constants_v2(func)
    frozen.graph.as_graph_def()

    return frozen.graph

def shuffle_split_data(data, labels, train_split):
    seed = random.randint(0, 256)
    data = tf.random.shuffle(tf.convert_to_tensor(data), seed=seed)
    labels = tf.random.shuffle(tf.convert_to_tensor(labels), seed=seed)

    split_idx = round(len(data) * train_split)
    return data[:split_idx], labels[:split_idx], data[split_idx:], labels[split_idx:]

def train():
    images, mappings, labels = inout.from_cdx_file(config.INPUT if config.INPUT else inout.get_input('Path to CDX: '))
    images = [image / 255.0 for image in images]

    try: train_split = float(config.TRAIN_SPLIT if config.TRAIN_SPLIT else inout.get_input('Percent train split [float 0-100]: ')) / 100
    except ValueError: inout.err('Invalid train split. Must be float.')
    if not 0 <= train_split <= 100: inout.err('Invalid train split. Must be between 0 and 100 inclusive.')

    try: epochs = int(config.EPOCHS if config.EPOCHS else inout.get_input('Epochs [integer > 0]: '))
    except ValueError: inout.err('Invalid epoch count. Must be integer.')
    if epochs < 1: inout.err('Invalid epoch count. Must be positive integer.')

    try: required_accuracy = float(config.REQUIRED_ACCURACY if config.REQUIRED_ACCURACY else 0)
    except ValueError: inout.err('Invalid iteration count. Must be float.')

    layers, units_list, activations = [], [], []

    while True:
        draw_model_summary(layers, units_list, activations)

        new_layer = inout.get_input('New layer type (leave blank if done): ').lower()
        if not new_layer: break

        elif new_layer == 'dense':
            layers.append('Dense')

            try: units_list.append(int(inout.get_input('Units [integer > 0]: ')))
            except ValueError: inout.err('Invalid units.')
            if units_list[-1] <= 0: inout.err('Invalid units.')

            activations.append(inout.get_input('Activation: '))
        elif new_layer == 'flatten':
            layers.append('Flatten')
            units_list.append('N/A')
            activations.append('N/A')
        elif new_layer in ['convolution1d', 'conv1d']:
            layers.append('Convolution1D')

            try: units_list.append(int(inout.get_input('Units [integer > 0]: ')))
            except ValueError: inout.err('Invalid units.')
            if units_list[-1] <= 0: inout.err('Invalid units.')

            activations.append(inout.get_input('Activation: '))
        else: inout.err('Invalid layer type.')

    if len(layers) == 0: inout.err('Must have at least one layer.')

    layers.append('Dense')
    units_list.append(len(mappings))
    activations.append(inout.get_input('Output layer activation: '))
    draw_model_summary(layers, units_list, activations)

    model = tf.keras.models.Sequential()

    for i in range(len(layers)):
        layer = layers[i]
        units = units_list[i]
        activation = activations[i]

        layer_str = f'tf.keras.layers.{layer}('
        if units != 'N/A': layer_str += f'{units},'
        if activation != 'N/A': layer_str += f'activation="{activation}",'
        if i == 0: layer_str += 'input_shape=train_data.shape[1:]'
        layer_str += ')'

        model.add(eval(layer_str))

    optimizer = inout.get_input('Optimizer: ')
    loss_metric = inout.get_input('Loss metric: ').replace('_', '').lower()
    if loss_metric in ['sparsecategoricalcrossentropy', 'scc']: loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss_metric, metrics=['accuracy'])

    accuracy = 0
    while accuracy < required_accuracy:
        train_data, train_labels, test_data, test_labels = shuffle_split_data(images, labels, train_split)

        print()
        model.fit(train_data, train_labels, epochs=epochs)
        loss, accuracy = model.evaluate(test_data, test_labels, verbose=False)

        print()
        print(f'Testing Accuracy: {accuracy}')
        print(f'Testing Loss: {loss}')

    save_path = config.OUTPUT if config.OUTPUT else inout.get_input('Path to save model (leave blank to discard): ')
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