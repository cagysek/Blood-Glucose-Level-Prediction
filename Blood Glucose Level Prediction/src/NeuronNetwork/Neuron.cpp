//
//  Neuron.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#include "Layer.hpp"
#include <iostream>

double Neuron::ETA = 0.5;
double Neuron::ALPHA = 0.5;

Neuron::Neuron(unsigned numberOfOutputs, unsigned layerNeuronIndex)
{
    for (unsigned i = 0 ; i < numberOfOutputs ; i++)
    {
        outputWeights.push_back(Connection());
    }
    
    neuronIndex = layerNeuronIndex;
}

void Neuron::feedForward(Layer &prevLayer)
{
    double sum = 0.0;
    
    // prodju výstupy předešlé vrstvy a vynásobím s vahami propojení
    for (unsigned i = 0 ; i < prevLayer.getNeuronCount() ; i++)
    {
        
        sum += prevLayer.getNeuron(i).getOutputValue()
                * prevLayer.getNeuron(i).getNeuronOutputWeight(neuronIndex);
        
    }
    
    output = activationFunction(sum);
}

/**
    Aktivační funkce, použito TanH
 */
double Neuron::activationFunction(double x)
{
    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x)
{
    return 1.0 - (tanh(x) * tanh(x));
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - output;
    gradient = delta * Neuron::activationFunctionDerivative(output);
}

void Neuron::calcHiddenGradients(Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    gradient = dow * Neuron::activationFunctionDerivative(output);
}

double Neuron::sumDOW(Layer &nextLayer)
{
    double sum = 0.0;
    
    for (unsigned n = 0 ; n < nextLayer.getNeuronCount() - 1 ; n++)
    {
        sum += outputWeights[n].weight * nextLayer.getNeuron(n).gradient;
    }
    
    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    for (unsigned n = 0 ; n < prevLayer.getNeuronCount() ; n++)
    {
        Neuron& neuron = prevLayer.getNeuron(n);
        double oldDeltaWeight = neuron.getNeuronOutputDeltaWeight(neuronIndex);
        
        double newDeltaWeight =
                ETA // training RATE
                * neuron.getOutputValue()
                * gradient
                + ALPHA // momentum
                * oldDeltaWeight;
        
        //neuron.updateConnectionValues(newDeltaWeight);
        neuron.outputWeights[neuronIndex].weight += newDeltaWeight;
        neuron.outputWeights[neuronIndex].deltaWeight = newDeltaWeight;
    }
    
    
}

void Neuron::updateConnectionValues(double deltaWeight)
{
    outputWeights[neuronIndex].weight = outputWeights[neuronIndex].weight + deltaWeight;
    outputWeights[neuronIndex].deltaWeight = deltaWeight;
}
