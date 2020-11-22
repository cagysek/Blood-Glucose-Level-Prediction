//
//  Neuron.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#include "Layer.hpp"
#include <iostream>

double Neuron::ETA = 0.1;
double Neuron::ALPHA = 0.8;

Neuron::Neuron(unsigned number_of_outputs, unsigned layer_neuron_index)
{
    for (unsigned i = 0 ; i < number_of_outputs ; i++)
    {
        m_output_weights.push_back(Connection());
    }
    
    m_neuronIndex = layer_neuron_index;
}

void Neuron::feed_forward_hidden(Layer &prev_layer)
{
    double sum = 0.0;
    
    // prodju výstupy předešlé vrstvy a vynásobím s vahami propojení
    for (unsigned i = 0 ; i < prev_layer.get_neuron_count() ; i++)
    {
        double value = prev_layer.get_neuron(i).get_output_value()
        * prev_layer.get_neuron(i).get_neuron_output_weight(m_neuronIndex);
        
        sum += value;
        
        prev_layer.get_neuron(i).increase_weight_counter(m_neuronIndex, value);
        
    }
    
    m_output = activation_function_tanh(sum);
}

/**
    Spočte hodnoty na výstupu bez použití aktivační funkce
 */
void Neuron::feed_forward_output(Layer &prev_layer)
{
    double sum = 0.0;
    
    // prodju výstupy předešlé vrstvy a vynásobím s vahami propojení
    for (unsigned i = 0 ; i < prev_layer.get_neuron_count() ; i++)
    {
        double value = prev_layer.get_neuron(i).get_output_value()
        * prev_layer.get_neuron(i).get_neuron_output_weight(m_neuronIndex);
        
        sum += value;
        
        prev_layer.get_neuron(i).increase_weight_counter(m_neuronIndex, value);
        
    }
    
    m_output = sum;
}

void Neuron::apply_sigmoid_function(double sum_exp)
{
    m_output = exp(m_output) / sum_exp;
}

/**
    Aktivační funkce, použito TanH
 */
double Neuron::activation_function_tanh(double x)
{
    return tanh(x);
}

double Neuron::activation_function_softmax(double x)
{
    
    return 0.0;
}

double Neuron::activation_function_derivative(double x)
{
    return 1.0 - (tanh(x) * tanh(x));
}

void Neuron::calc_output_gradients(double target_val)
{
    double delta = target_val - m_output;
    m_gradient = delta * Neuron::activation_function_derivative(m_output);
}

void Neuron::calc_hidden_gradients(Layer &next_layer)
{
    double dow = sum_dow(next_layer);
    m_gradient = dow * Neuron::activation_function_derivative(m_output);
}

double Neuron::sum_dow(Layer &next_layer)
{
    double sum = 0.0;
    
    for (unsigned n = 0 ; n < next_layer.get_neuron_count() - 1 ; n++)
    {
        sum += m_output_weights[n].weight * next_layer.get_neuron(n).m_gradient;
    }
    
    return sum;
}

void Neuron::update_input_weights(Layer &prev_layer)
{
    // procházím předešlou vrstvu
    for (unsigned n = 0 ; n < prev_layer.get_neuron_count() ; n++)
    {
        // beru průběžně neurony a jejich spojení v aktuálním neuronem (m_neuronIndex)
        Neuron& neuron = prev_layer.get_neuron(n);
        double old_delta_weight = neuron.get_neuron_output_delta_weight(m_neuronIndex);
        
        double new_delta_weight =
                ETA // training RATE
                * neuron.get_output_value()
                * m_gradient
                + ALPHA // momentum
                * old_delta_weight;
        
        //neuron.updateConnectionValues(newDeltaWeight);
        neuron.m_output_weights[m_neuronIndex].weight += new_delta_weight;
        neuron.m_output_weights[m_neuronIndex].delta_weight = new_delta_weight;
    }
    
    
}

void Neuron::increase_weight_counter(unsigned weight_index, double transmitted_val)
{
    m_output_weights[weight_index].transmitted_value_counter += transmitted_val;
    m_output_weights[weight_index].last_transmitted_value = transmitted_val;
}

void Neuron::increase_weight_counter_error(unsigned weight_index)
{
    m_output_weights[weight_index].transmitted_value_relative_error_counter += m_output_weights[weight_index].last_transmitted_value;
}

void Neuron::update_connection_values(double delta_weight)
{
    m_output_weights[m_neuronIndex].weight = m_output_weights[m_neuronIndex].weight + delta_weight;
    m_output_weights[m_neuronIndex].delta_weight = delta_weight;
}


