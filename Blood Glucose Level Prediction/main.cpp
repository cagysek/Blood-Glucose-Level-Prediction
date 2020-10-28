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
#include "Constants.h"

void showVectorVals(const std::string& label, const std::vector<double> &v)
{
    
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << std::ceil(v[i] * 100.0) / 100.0 << " ";
    }

    std::cout << std::endl;
}


void prepare_target_values(std::vector<double> &target_values, const double prediction_value)
{
    target_values.clear();
    
    // vytvořím pole, které představuje výstup, defaultně dám všude 0
    for (unsigned i = 0 ; i < Constants::Internal_Bound_Count ; i++)
    {
        target_values.push_back(-1);
    }
    
    // z průměrný hodnoty predikce pro hodnoty získám index na výstup (co by to mělo ukazovat)
    unsigned prediction_index = Constants::Level_To_Index_Band(prediction_value);
    
    // nasadím hodnotu predikovaného výstupu na 1
    target_values[prediction_index] = 1;
  
}




int main(int argc, const char * argv[]) {
    
    Data_reader data_reader("/Users/cagy/Documents/Škola/PPR/Blood-Glucose-Level-Prediction/Blood Glucose Level Prediction/data/asc2018.sqlite");
    
    std::vector<double> input_values;
    std::vector<double> target_values;
    std::vector<double> prediction_values;
    
    std::vector<Segment> segments;
    
    // predikce na 60 min
    unsigned prediction_for = 60 / 5; // intervali jsou po 5 min, tímhle zjistím o kolik se posunout
    
    std::vector<unsigned> topology;
    topology.push_back(8);
    topology.push_back(16);
    topology.push_back(26);
    topology.push_back(32);
    
    Neuron_network neuronNetwork(topology);
    
    int segment_id = 0;
    int offset = 0;
    int limit = 8;
    
    data_reader.open();
    
    data_reader.init_segments(segments);
    
    double min_error = 100.0;
    
    int k = 0;
    
    int segment_counter = 0;
    
    int counter = 1;
    
    int target_offset = 0;
    
    // první init parametrů
    segment_id = segments[segment_counter].m_segment_id;
    offset = segments[segment_counter].m_start_id;
    
    int count = 0;
    
    while (true)
    {
        count = data_reader.get_input_data(input_values, limit, offset, segment_id);
        
        // posunu se o zadanej čas o kolik chci predikovat
        target_offset = offset + prediction_for;
        
        // pokud nemám dost hodnot na vstupu
        if (input_values.size() != 8)
        {
            
            std::cout << "err 1 " << input_values.size() << std::endl;
            
            // pokud nejsou už další segmenty
            if (segments.size() < segment_counter + 1)
            {
                break;
            }
            
            segment_counter++;
            
            // nastavím offset + segmentId pro další selecty
            offset = segments[segment_counter].m_start_id;
            segment_id = segments[segment_counter].m_segment_id;
            
            continue;
        }
        
        std::cout << " " << std::endl;
        std::cout << counter << " NEW BATCH!!" << std::endl;
        for (int i = 0; i < input_values.size(); i++) {
            std::cout << input_values[i] << std::endl;
        }
        
        
        
        data_reader.get_prediction_data(prediction_values, limit, target_offset, segment_id);
        
        // pokud nemám k 8 datům 8 předpovědí, jedu dál
        if (prediction_values.size() != 8)
        {
            offset = offset + 8;
            continue;
        }
        
        std::cout << " " << std::endl;
        std::cout << counter << " OUT NEW BATCH!!" << std::endl;
        showVectorVals("Prediction", prediction_values);
        double prediction_average = 0;
        for (int i = 0; i < prediction_values.size(); i++) {
            prediction_average += prediction_values[i];
        }
        
        prediction_average /= prediction_values.size();
        
        prepare_target_values(target_values, prediction_average);
        
        offset = offset + limit;
     
        counter++;
        
        
        
        neuronNetwork.feed_forward_propagation(input_values);
        
        
        std::vector<double> resultValues;
        neuronNetwork.get_results(resultValues);
        
        showVectorVals("Result:", resultValues);
        
        
        neuronNetwork.back_propagation(target_values);
        
        showVectorVals("Targets:", target_values);
        
        
        // Report how well the training is working, average over recent samples:
                std::cout << "Net recent average error: "
                        << neuronNetwork.get_recent_average_error() << std::endl;
        
        if (min_error > neuronNetwork.get_error())
        {
            min_error = neuronNetwork.get_error();
        }
        
        std::cout << "Net error: "
                << neuronNetwork.get_error() << std::endl;
        
    }
        
       
    
    
    std::cout << "min error: "
            << min_error << std::endl;
 //   data_reader.close();
    
    
    
    
    std::cout << "Konec" << std::endl;
    
    return 0;
}

