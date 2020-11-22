//
//  Neuron.hpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#ifndef Neuron_hpp
#define Neuron_hpp

#include <stdio.h>
#include <vector>
#include <cmath>
#include "Connection.hpp"

class Layer;

class Neuron
{
public:
    Neuron(unsigned number_of_outputs, unsigned neuron_index);
    void set_output_value(double new_output_val) { m_output = new_output_val; };
    double get_output_value() { return m_output; }
    void feed_forward_hidden(Layer &prev_layer);
    void feed_forward_output(Layer &prev_layer);
    void apply_sigmoid_function(double sum_exp);
    double get_neuron_output_weight(unsigned index) { return m_output_weights[index].weight; }
    double get_neuron_output_delta_weight(unsigned index) { return m_output_weights[index].delta_weight; }
    void calc_output_gradients(double target_Val);
    void calc_hidden_gradients(Layer &next_layer);
    void update_input_weights(Layer &prev_layer);
    void update_connection_values(double delta_weight);
    
    void increase_weight_counter(unsigned index, double transmitted_val);
    void increase_weight_counter_error(unsigned index);
    
    static double ETA; // [0.0 - 1.0] - konstanta jak moc se má síť učit, "jak moc aktuální krok ovlivní další krok"
    static double ALPHA; // momentum, násobič poslední změny váhy "jak moc minulá změna ovlivní nový krok"
    
    std::vector<Connection> const get_weights() { return m_output_weights; }
    Connection const get_weight(unsigned index) { return m_output_weights[index]; }
    
    unsigned m_neuronIndex;
    double m_gradient;
private:
    double m_output;
    
    std::vector<Connection> m_output_weights;
    
    static double activation_function_tanh(const double x);
    static double activation_function_softmax(double x);
    static double activation_function_derivative(const double x);
    
    double sum_dow(Layer &next_layer);
    
    

};



#endif /* Neuron_hpp */
