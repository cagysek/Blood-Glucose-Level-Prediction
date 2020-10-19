//
//  main.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#include <iostream>
#include <vector>

#include "Neuron_network.hpp"
#include "Data_reader.hpp"

void showVectorVals(const std::string& label, const std::vector<double> &v)
{
    
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << std::ceil(v[i] * 100.0) / 100.0 << " ";
    }

    std::cout << std::endl;
}


int main(int argc, const char * argv[]) {
    
    Data_reader data_reader("/Users/cagy/Documents/Škola/PPR/Blood-Glucose-Level-Prediction/Blood Glucose Level Prediction/data/asc2018.sqlite");
    data_reader.open();
    std::vector<double> inputValues;
    data_reader.get_input_data(inputValues, 0, 8);
    
    data_reader.close();
 //   return 0;
    std::vector<unsigned> topology;
    topology.push_back(8);
    topology.push_back(16);
    topology.push_back(26);
    topology.push_back(32);
    
    Neuron_network neuronNetwork(topology);
    

    inputValues.push_back(11.05);
    inputValues.push_back(10.88);
    inputValues.push_back(10.88);
    inputValues.push_back(10.88);
    inputValues.push_back(10.77);
    inputValues.push_back(10.66);
    inputValues.push_back(10.49);
    inputValues.push_back(10.32);
  
    
    
    
    
    
    unsigned i = 0;
    
    while (true) {
        showVectorVals("Inputs:", inputValues);
        neuronNetwork.feed_forward_propagation(inputValues);
        
        
        
        std::vector<double> resultValues;

        neuronNetwork.get_results(resultValues);
        showVectorVals("Outputs:", resultValues);
        
        std::vector<double> targetValues;
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(1);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
   
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        
        
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        targetValues.push_back(0);
        
        targetValues.push_back(1);
        targetValues.push_back(0);
        
        showVectorVals("Targets:", targetValues);
        
        
        neuronNetwork.back_propagation(targetValues);
        
        // Report how well the training is working, average over recent samples:
                std::cout << "Net recent average error: "
                        << neuronNetwork.get_recent_average_error() << std::endl;
        
        i++;
        
        if (i >= 1000)
        {
            break;
        }
    }
    
    
    
    
    
    std::cout << "Konec" << std::endl;
    
    return 0;
}

