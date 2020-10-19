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
        void addNeuron(Neuron neuron);
        Neuron& getNeuron(unsigned index);
        unsigned getNeuronCount();
        Neuron& getBias();
    
    private:
        std::vector<Neuron> neurons;
};

#endif /* Layer_hpp */
