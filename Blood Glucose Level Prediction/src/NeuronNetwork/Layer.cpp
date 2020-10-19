//
//  Layer.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#include "Layer.hpp"


void Layer::addNeuron(Neuron neuron)
{
    neurons.push_back(neuron);
}

Neuron& Layer::getNeuron(unsigned index)
{
    return neurons[index];
}

unsigned Layer::getNeuronCount()
{
    return (unsigned)neurons.size();
}

Neuron& Layer::getBias()
{
    return neurons.back();
}
