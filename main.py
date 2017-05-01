import json
import math


with open('data.json') as data_file:
    data_dict = json.load(data_file)


inputs = data_dict['inputs']
weights_to_image_layer = data_dict['weights_to_image_layer']
count_of_neurons_in_image_layer = data_dict['count_of_neurons_in_image_layer']
radius_of_gauss_function = data_dict['radius_of_gauss_function']

y = {}
theta = {}

for image in count_of_neurons_in_image_layer.keys():
    y[image] = []
    for image_neuron in range(0, count_of_neurons_in_image_layer[image]):
        weighted_sum = 0.0
        for input_neuron in range(0, len(inputs)):
            t = (weights_to_image_layer[input_neuron][image][image_neuron] - inputs[input_neuron])
            weighted_sum += math.exp(
                -t * t / radius_of_gauss_function * radius_of_gauss_function
            )
        y[image].append(weighted_sum)

for image in count_of_neurons_in_image_layer.keys():
    image_sum = 0.0
    for image_neuron in range(0, count_of_neurons_in_image_layer[image]):
        image_sum += y[image][image_neuron]
    image_sum /= count_of_neurons_in_image_layer[image]
    theta[image] = image_sum

best_result = max(theta.values())

for k in theta.keys():
    if theta[k] == best_result:
        print(k)
