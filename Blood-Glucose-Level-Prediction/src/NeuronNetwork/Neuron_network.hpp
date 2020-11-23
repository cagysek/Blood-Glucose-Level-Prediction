//
//  NeuronNetwork.hpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#ifndef Neuron_network_hpp
#define Neuron_network_hpp

#include <stdio.h>
#include <vector>
#include "Layer.hpp"
#include "Neuron.hpp"
#include "Constants.h"


class Neuron_network
{
    public:
        Neuron_network();
        Neuron_network(const std::vector<unsigned> &topology);
    
        void feed_forward_propagation(const std::vector<double> &input_values, bool use_backpropagation);
        void back_propagation(const std::vector<double> &target_values, double prediction_value);
        void count_error(const std::vector<double> &target_values, double prediction_value);
        void get_results(std::vector<double> &result_values);
        double risk_function(const double x);
    
        double get_error(void) const { return m_error; }
    
        double get_average_error(void);
        double get_stanadrd_deviation(void);
        std::vector<Layer> get_layers() const { return m_layers; };
        Layer& get_layer(int index) { return m_layers[index]; };
        std::vector<double> get_errors() const { return m_relative_error; };
        
    
    private:
        std::vector<Layer> m_layers; // [layerNumber][neuronNumber]
        double m_error;

    
        std::vector<double> m_relative_error;
    
        void save_transmitted_value_error();
};


#endif /* NeuronNetwork_hpp */
