//
//  NeuronNetwork.hpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#ifndef NeuronNetwork_hpp
#define NeuronNetwork_hpp

#include <stdio.h>
#include <vector>
#include "Layer.hpp"
#include "Neuron.hpp"


class NeuronNetwork
{
    public:
        NeuronNetwork(const std::vector<unsigned> &topology);
        void feedForwardPropagation(const std::vector<double> &inputValues);
        void backPropagation(const std::vector<double> &targetValues);
        void getResults(std::vector<double> &resultValues);
        double riskFunction(const double x);
    
    static double m_recentAverageSmoothingFactor;
    double getRecentAverageError(void) const { return m_recentAverageError; }
    
    private:
        std::vector<Layer> layers; // [layerNumber][neuronNumber]
        double error;
        double m_recentAverageError;
    
};


#endif /* NeuronNetwork_hpp */
