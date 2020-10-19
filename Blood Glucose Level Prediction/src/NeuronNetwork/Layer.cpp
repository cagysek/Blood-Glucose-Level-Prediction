//
//  Layer.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#include "Layer.hpp"


void Layer::add_neuron(Neuron neuron)
{
    m_neurons.push_back(neuron);
}

Neuron& Layer::get_neuron(unsigned index)
{
    return m_neurons[index];
}

unsigned Layer::get_neuron_count()
{
    return (unsigned)m_neurons.size();
}

Neuron& Layer::get_bias()
{
    return m_neurons.back();
}
