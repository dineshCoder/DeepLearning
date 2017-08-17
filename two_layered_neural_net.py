from single_layer import single_neutron
import numpy as np
import logging
logger = logging.getLogger(__name__)


class NeuralNetwork:
    """
    Implement a Neural network of given required shape.
    It implements both forward propogation and Backward propogation.
    """
    def __init__(self, number_of_layers_neuron_dictionary):
        self.number_of_layers = len(number_of_layers_neuron_dictionary)

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass
