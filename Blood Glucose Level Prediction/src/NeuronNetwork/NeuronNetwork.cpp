//
//  NeuronNetwork.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#include "NeuronNetwork.hpp"
#include <iostream>

double NeuronNetwork::m_recentAverageSmoothingFactor = 100.0;

NeuronNetwork::NeuronNetwork(const std::vector<unsigned> &topology)
{
    unsigned numberOfLayers = topology.size();
    
    // projdu předaný pole, kde mám strukturu
    for (unsigned layerNum = 0 ; layerNum < numberOfLayers ; layerNum++)
    {
        // přidáme novou vrstvu
        layers.push_back(Layer());
        
        // zjistím počet výstupů = počet neuronů v další vrstvě
        unsigned numberOfOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        
        // vytvořím jednotlivý neurony ve vrstvách + přidám bias
        for (unsigned neuronNum = 0 ; neuronNum <= topology[layerNum] ; neuronNum++)
        {
            layers.back().addNeuron(Neuron(numberOfOutputs, neuronNum));
        }
        
        // nastavení váhy bias
        layers.back().getBias().setOutputValue(1.0);
    }
}


void NeuronNetwork::feedForwardPropagation(const std::vector<double> &inputValues)
{
    for (unsigned i = 0 ; i < inputValues.size() ; i++)
    {
        layers[0].getNeuron(i).setOutputValue(riskFunction(inputValues[i]));
    }
    
    for (unsigned layerNum = 1 ; layerNum < layers.size() ; layerNum++)
    {
        Layer &prevLayer = layers[layerNum - 1];

        for (unsigned neuronNum = 0 ; neuronNum < layers[layerNum].getNeuronCount() - 1 ; neuronNum++)
        {
            layers[layerNum].getNeuron(neuronNum).feedForward(prevLayer);
        }
    }
}

void NeuronNetwork::backPropagation(const std::vector<double> &targetValues)
{
    // RMS
    Layer &outputLayer = layers.back();
    error = 0.0;
    
    for (unsigned i = 0 ; i < outputLayer.getNeuronCount() - 1 ; i++)
    {
        double delta = targetValues[i] - outputLayer.getNeuron(i).getOutputValue();
        error += delta * delta;
    }
    
    error /= outputLayer.getNeuronCount() - 1;
    error = sqrt(error); // RMS
    
    m_recentAverageError =
                (m_recentAverageError * m_recentAverageSmoothingFactor + error)
                / (m_recentAverageSmoothingFactor + 1.0);

    
    // gradient výstup
    for (unsigned i = 0 ; i < outputLayer.getNeuronCount() - 1 ; i++)
    {
        Neuron& neuron = outputLayer.getNeuron(i);
        neuron.calcOutputGradients(targetValues[i]);
    }
    
    
    // gradient na hidden layers
    for (unsigned layerNum = layers.size() - 2 ; layerNum > 0 ; layerNum--)
    {
        Layer &hiddenLayer = layers[layerNum];
        Layer &nextLayer = layers[layerNum + 1];
        
        for (unsigned neuronNum = 0 ; neuronNum < hiddenLayer.getNeuronCount() ; neuronNum++)
        {
            Neuron& neuron = hiddenLayer.getNeuron(neuronNum);
            neuron.calcHiddenGradients(nextLayer);
        }
    }
    
    //
    
    // update vah
    
    for (unsigned layerNum = layers.size() - 1 ; layerNum > 0 ; layerNum--)
    {
        Layer &layer = layers[layerNum];
        Layer &prevLayer = layers[layerNum - 1];
        
        for (unsigned neuronNum = 0 ; neuronNum < layer.getNeuronCount() - 1 ; neuronNum++)
        {
            Neuron& neuron = layer.getNeuron(neuronNum);
            neuron.updateInputWeights(prevLayer);
        }
        
    }
    
}

void NeuronNetwork::getResults(std::vector<double> &resultValues)
{
    resultValues.clear();

    for (unsigned n = 0; n < layers.back().getNeuronCount() - 1; ++n) {
        resultValues.push_back(layers.back().getNeuron(n).getOutputValue());
    }
}

double NeuronNetwork::riskFunction(const double x)
{
    // DOI:  10.1080/10273660008833060
    const double original_risk = 1.794 * (pow(log(x), 1.026) - 1.861);    //mmol/L
        
    return original_risk / 3.5;
}
