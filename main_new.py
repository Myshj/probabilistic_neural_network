import json
import math


# открыли файл настроек
with open('settings.json') as settings_file:
    settings_dict = json.load(settings_file)
if not isinstance(settings_dict, dict):
    raise ValueError('Dict expected')

count_of_input_neurons = settings_dict['count_of_input_neurons']
count_of_neurons_in_image_layer = settings_dict['count_of_neurons_in_image_layer']
radius_of_gauss_function = settings_dict['radius_of_gauss_function']


# открыли файл с обучающими примерами данными
with open('learning_examples.json') as learning_examples__file:
    learning_examples_dict = json.load(learning_examples__file)
if not isinstance(learning_examples_dict, dict):
    raise ValueError("Dict expected")


# создали таблицу весов от входного слоя к слою образов
weights_to_image_layer = [{} for i in range(0, count_of_input_neurons)]
for image, examples in learning_examples_dict.items():
    # для каждого входного нейрона подготовили массив его весов
    for input_neuron_weights in weights_to_image_layer:
        input_neuron_weights[image] = []

    # запомнили все примеры, связанные с образом
    for example in examples:
        for input_neuron, weight in enumerate(example):
            weights_to_image_layer[input_neuron][image].append(weight)


# сохранили таблицу весов
with open('weights_to_image_layer.json', 'w') as weights_file:
    json.dump(weights_to_image_layer, weights_file, indent=4)




# открыли файл с входами
with open('inputs.json') as inputs_file:
    inputs = json.load(inputs_file)
if not isinstance(inputs, list):
    raise ValueError('List expected')

# для хранения выхода слоёв
y = {}
theta = {}

# считаем выходы слоя образов
for image in count_of_neurons_in_image_layer.keys():
    y[image] = []
    for image_neuron in range(0, count_of_neurons_in_image_layer[image]):
        weighted_sum = 0.0
        for input_neuron in range(0, count_of_input_neurons):
            t = (weights_to_image_layer[input_neuron][image][image_neuron] - inputs[input_neuron])
            weighted_sum += math.exp(
                -t * t / (radius_of_gauss_function * radius_of_gauss_function)
            )
        y[image].append(weighted_sum)

# считаем выходы слоя сложения
for image in count_of_neurons_in_image_layer.keys():
    image_sum = 0.0
    for image_neuron in range(0, count_of_neurons_in_image_layer[image]):
        image_sum += y[image][image_neuron]
    image_sum /= count_of_neurons_in_image_layer[image]
    theta[image] = image_sum


# сохраняем выход сети
with open('outputs.json', 'w') as outputs_file:
    json.dump(theta, outputs_file)
