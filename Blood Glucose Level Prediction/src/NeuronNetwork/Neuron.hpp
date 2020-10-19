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
    Neuron(unsigned numberOfOutputs, unsigned neuronIndex);
    void setOutputValue(double newOutputVal) { output = newOutputVal; };
    double getOutputValue() { return output; }
    void feedForward(Layer &prevLayer);
    double getNeuronOutputWeight(unsigned index) { return outputWeights[index].weight; }
    double getNeuronOutputDeltaWeight(unsigned index) { return outputWeights[index].deltaWeight; }
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
    void updateConnectionValues(double deltaWeight);
    
    static double ETA; // [0.0 - 1.0] - konstanta jak moc se má síť učit
    static double ALPHA; // momentum, násobič poslední změny váhy
    unsigned neuronIndex;
    double gradient;
    
private:
    double output;
    std::vector<Connection> outputWeights;
    
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    
    double sumDOW(Layer &nextLayer);
    
    

};



#endif /* Neuron_hpp */
