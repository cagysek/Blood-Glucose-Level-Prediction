//
//  NeuronNetwork.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#include <iostream>

#include "Neuron_network.hpp"

#define GRAPH_ERROR_MAX_VAL 0.15

Neuron_network::Neuron_network(){}

Neuron_network::Neuron_network(const std::vector<unsigned> &topology)
{
    unsigned number_of_layers = (int)topology.size();
    
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
    
    m_error = -1;
}


void Neuron_network::feed_forward_propagation(const std::vector<double> &input_values, bool use_backpropagation)
{
    // na vstup přiřadím vstupní hodnoty
    for (unsigned i = 0 ; i < input_values.size() ; i++)
    {
        m_layers[0].get_neuron(i).set_output_value(risk_function(input_values[i]));
    }
    
    for (unsigned layer_num = 1 ; layer_num < m_layers.size() ; layer_num++)
    {
        Layer &prevLayer = m_layers[layer_num - 1];

        
            for (unsigned neuron_num = 0 ; neuron_num < m_layers[layer_num].get_neuron_count() - 1 ; neuron_num++)
            {
                // pokud se nejedná o výstup aplikuju tanH
                if (layer_num != m_layers.size())
                {
                    m_layers[layer_num].get_neuron(neuron_num).feed_forward_hidden(prevLayer);
                }
                // pokud je výstup, sečtu akorát hrany bez aktivační funkce, sigmoida se udělá až po cyklu
                else
                {
                    m_layers[layer_num].get_neuron(neuron_num).feed_forward_output(prevLayer);
                }
            }
    }
    
    
    // aplikace sigmoidy na výstup
    double sum_exp = 0.0;
    int max_index = 0;
    
    Layer output_layer = m_layers[m_layers.size() - 1];
    
    // procházím výstupy bez bias pro získání čitatele v sigmoide
    for (unsigned i = 0 ; i < output_layer.get_neuron_count() - 1 ; i++)
    {
        sum_exp += exp(output_layer.get_neuron(i).get_output_value());
    }
    
    // aplikuju sigmoidu na výstupy
    for (unsigned neuron_num = 0 ; neuron_num < output_layer.get_neuron_count() - 1 ; neuron_num++)
    {
        output_layer.get_neuron(neuron_num).apply_sigmoid_function(sum_exp);
        
        if (!use_backpropagation)
        {
            if (output_layer.get_neuron(max_index).get_output_value() < output_layer.get_neuron(neuron_num).get_output_value())
            {
                max_index = neuron_num;
            }
        }
    }
    
    // pokud false znamená to že se načetly hrany ze souboru
    if (!use_backpropagation)
    {
        printf("%f\n", Constants::Band_Index_To_Level(max_index));
    }
    
}

void Neuron_network::back_propagation(const std::vector<double> &target_values, double prediction_value)
{
    
    // RMS
    Layer &output_layer = m_layers.back();
    
    count_error(target_values, prediction_value);
    
    // vypočítá garient výstupů
    for (unsigned i = 0 ; i < output_layer.get_neuron_count() - 1 ; i++)
    {
        Neuron& neuron = output_layer.get_neuron(i);
        neuron.calc_output_gradients(target_values[i]);
    }
    
    
    // gradient na hidden layers
    for (unsigned layer_num = (int)m_layers.size() - 2 ; layer_num > 0 ; layer_num--)
    {
        Layer &hidden_layer = m_layers[layer_num];
        Layer &next_layer = m_layers[layer_num + 1];
        
        for (unsigned neuron_num = 0 ; neuron_num < hidden_layer.get_neuron_count() ; neuron_num++)
        {
            Neuron& neuron = hidden_layer.get_neuron(neuron_num);
            neuron.calc_hidden_gradients(next_layer);
        }
    }
    
    // update vah
    
    for (unsigned layer_num = (int)m_layers.size() - 1 ; layer_num > 0 ; layer_num--)
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

void Neuron_network::count_error(const std::vector<double> &target_values, double prediction_value)
{
    Layer &output_layer = m_layers.back();
    
    m_error = 0.0;
    
    unsigned biggest_value_index = 0;
    
    for (unsigned i = 0 ; i < output_layer.get_neuron_count() - 1 ; i++)
    {
        double output_value = output_layer.get_neuron(i).get_output_value();
        
        double delta = target_values[i] - output_value;
        m_error += delta * delta;
        
        if (output_value > output_layer.get_neuron(biggest_value_index).get_output_value())
        {
            biggest_value_index = i;
        }
    }
    
    // relativní error
    double calculated_prediction = Constants::Band_Index_To_Level(biggest_value_index);
    
    double relative_error = abs(calculated_prediction - prediction_value) / prediction_value;
    
    m_relative_error.push_back(relative_error);
    
    if (relative_error < GRAPH_ERROR_MAX_VAL)
    {
        save_transmitted_value_error();
    }
    
    // error pro backpropagation
    m_error /= output_layer.get_neuron_count() - 1;
    m_error = sqrt(m_error); // RMS
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

double Neuron_network::get_average_error()
{
    double sum = 0.0;
    
    for (unsigned i = 0; i < m_relative_error.size() ; i++)
    {
        sum += m_relative_error[i];
    }
    
    return sum / m_relative_error.size();
}

double Neuron_network::get_stanadrd_deviation()
{
    double relative_error_mean = get_average_error();
    
    double sum = 0.0;
    
    for (unsigned i = 0; i < m_relative_error.size() ; i++)
    {
        sum += pow(m_relative_error[i] - relative_error_mean, 2);
    }
    
    return sqrt(sum / (m_relative_error.size() - 1));
    
}

void Neuron_network::save_transmitted_value_error()
{
    for (unsigned i = 0; i < m_layers.size() - 1; i++) {
        for (unsigned j = 0; j < m_layers[i].get_neuron_count(); j++) {
            for (unsigned k = 0; k < m_layers[i].get_neuron_count(); k++) {
                m_layers[i].get_neuron(j).increase_weight_counter_error(k);
            }
        }
    }
}

