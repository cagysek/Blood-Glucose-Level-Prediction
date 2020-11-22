//
//  Layer.hpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <stdio.h>
#include <vector>


#include "Neuron.hpp"

class Layer
{
    public:
        void add_neuron(Neuron neuron);
        Neuron& get_neuron(unsigned index);
        unsigned get_neuron_count();
        Neuron& get_bias();
    
    private:
        std::vector<Neuron> m_neurons;
};

#endif /* Layer_hpp */
