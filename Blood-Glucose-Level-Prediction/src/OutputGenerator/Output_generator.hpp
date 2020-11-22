//
//  OutputGenerator.hpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 22.11.2020.
//

#ifndef Output_generator_hpp
#define Output_generator_hpp

#include <stdio.h>
#include "Neuron_network.hpp"
#include <fstream>
#include <iostream>
#include <limits.h>

class Output_generator
{
public:

    void generate_graph(const Neuron_network neural_network);
    void generate_init_file(const Neuron_network neural_network);
    
    void generate_graph_transmitted_values(const Neuron_network neural_network);
    void generate_graph_transmitted_values_error(const Neuron_network neural_network);
    void generate_error_csv(const Neuron_network neural_network);
    
private:
    void generate_graph(const Neuron_network neural_network, bool show_all_transmitted_values);
    
};

#endif /* Output_generator_hpp */
