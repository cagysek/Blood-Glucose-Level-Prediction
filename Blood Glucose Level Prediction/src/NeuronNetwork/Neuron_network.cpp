//
//  NeuronNetwork.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#include <iostream>

#include "Neuron_network.hpp"



double Neuron_network::m_recent_average_smoothing_factor = 100.0;

Neuron_network::Neuron_network(const std::vector<unsigned> &topology)
{
    unsigned number_of_layers = topology.size();
    
    // projdu předaný pole, kde mám strukturu
    for (unsigned layer_num = 0 ; layer_num < number_of_layers ; layer_num++)
    {
        // přidáme novou vrstvu
        m_layers.push_back(Layer());
        
        // zjistím počet výstupů = počet neuronů v další vrstvě
        unsigned number_of_outputs = layer_num == topology.size() - 1 ? 0 : topology[layer_num + 1];
        
        // vytvořím jednotlivý neurony ve vrstvách + přidám bias
        for (unsigned neuron_num = 0 ; neuron_num <= topology[layer_num] ; neuron_num++)
        {
            m_layers.back().add_neuron(Neuron(number_of_outputs, neuron_num));
        }
        
        // nastavení váhy bias
        m_layers.back().get_bias().set_output_value(1.0);
    }
}


void Neuron_network::feed_forward_propagation(const std::vector<double> &input_values)
{
    for (unsigned i = 0 ; i < input_values.size() ; i++)
    {
        m_layers[0].get_neuron(i).set_output_value(risk_function(input_values[i]));
    }
    
    for (unsigned layer_num = 1 ; layer_num < m_layers.size() ; layer_num++)
    {
        Layer &prevLayer = m_layers[layer_num - 1];

        for (unsigned neuron_num = 0 ; neuron_num < m_layers[layer_num].get_neuron_count() - 1 ; neuron_num++)
        {
            m_layers[layer_num].get_neuron(neuron_num).feed_forward(prevLayer);
        }
    }
}

void Neuron_network::back_propagation(const std::vector<double> &target_values)
{
    // RMS
    Layer &output_layer = m_layers.back();
    
    m_error = 0.0;
    
    for (unsigned i = 0 ; i < output_layer.get_neuron_count() - 1 ; i++)
    {
        double delta = target_values[i] - output_layer.get_neuron(i).get_output_value();
        m_error += delta * delta;
    }
    
    m_error /= output_layer.get_neuron_count() - 1;
    m_error = sqrt(m_error); // RMS
    
    m_recent_average_error =
                (m_recent_average_error * m_recent_average_smoothing_factor + m_error)
                / (m_recent_average_smoothing_factor + 1.0);

    
    // vypočítá garient výstupů
    for (unsigned i = 0 ; i < output_layer.get_neuron_count() - 1 ; i++)
    {
        Neuron& neuron = output_layer.get_neuron(i);
        neuron.calc_output_gradients(target_values[i]);
    }
    
    
    // gradient na hidden layers
    for (unsigned layer_num = m_layers.size() - 2 ; layer_num > 0 ; layer_num--)
    {
        Layer &hidden_layer = m_layers[layer_num];
        Layer &next_layer = m_layers[layer_num + 1];
        
        for (unsigned neuron_num = 0 ; neuron_num < hidden_layer.get_neuron_count() ; neuron_num++)
        {
            Neuron& neuron = hidden_layer.get_neuron(neuron_num);
            neuron.calc_hidden_gradients(next_layer);
        }
    }
    
    //
    
    // update vah
    
    for (unsigned layer_num = m_layers.size() - 1 ; layer_num > 0 ; layer_num--)
    {
        Layer &layer = m_layers[layer_num];
        Layer &prev_layer = m_layers[layer_num - 1];
        
        for (unsigned neuron_num = 0 ; neuron_num < layer.get_neuron_count() - 1 ; neuron_num++)
        {
            Neuron& neuron = layer.get_neuron(neuron_num);
            neuron.update_input_weights(prev_layer);
        }
        
    }
    
}

void Neuron_network::get_results(std::vector<double> &result_values)
{
    result_values.clear();

    for (unsigned n = 0; n < m_layers.back().get_neuron_count() - 1; ++n) {
        result_values.push_back(m_layers.back().get_neuron(n).get_output_value());
    }
}

double Neuron_network::risk_function(const double x)
{
    // DOI:  10.1080/10273660008833060
    const double original_risk = 1.794 * (pow(log(x), 1.026) - 1.861);    //mmol/L
        
    return original_risk / 3.5;
}
