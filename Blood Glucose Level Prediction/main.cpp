//
//  main.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#include <iostream>
#include <vector>
#include "tbb/parallel_for.h"
#include <mutex>

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


void prepare_target_values(const std::vector<double> &prediction_values, std::vector<double> &target_values)
{
    // vyčistím targets
    target_values.clear();
    
    // vypočitám průměr z předpovědí
    double prediction_average = 0;
    
    for (int i = 0; i < prediction_values.size(); i++) {
        prediction_average += prediction_values[i];
    }
    
    prediction_average /= prediction_values.size();
    
    // vytvořím vektor o velikosti výstupů, které představuje výstup, defaultně dám všude -1 (tanh <-1;1>)
    for (unsigned i = 0 ; i < Constants::Internal_Bound_Count ; i++)
    {
        target_values.push_back(-1);
    }
    
    // z průměrný hodnoty predikce pro hodnoty získám index na výstup (výstup, který by měl mít nejvyšší pravděpodobnost)
    unsigned prediction_index = Constants::Level_To_Index_Band(prediction_average);
    
    // nasadím hodnotu predikovaného výstupu na 1
    target_values[prediction_index] = 1;
  
}

void init_neuron_networks(std::vector<Neuron_network> &neuron_networks, const std::vector<unsigned> &topology)
{
    neuron_networks.clear();
    
    unsigned neuron_network_count = 20;
    
    for (unsigned i = 0 ; i < neuron_network_count ; i++)
    {
        neuron_networks.push_back(Neuron_network(topology));
    }
}

int main(int argc, const char * argv[]) {
    
    Data_reader data_reader("/Users/cagy/Documents/Škola/PPR/Blood-Glucose-Level-Prediction/Blood Glucose Level Prediction/data/asc2018.sqlite");
    
    // init základních vektorů
    std::vector<Neuron_network> neuron_networks;
    std::vector<double> input_values;
    std::vector<double> target_values;
    std::vector<double> prediction_values;
    
    std::vector<Segment> segments;
    
    // predikce na 60 min
    unsigned prediction_for = 60 / 5; // intervali jsou po 5 min, tímhle zjistím o kolik se posunout
    
    // vytvoření topologie
    std::vector<unsigned> topology;
    topology.push_back(8);
    topology.push_back(16);
    topology.push_back(26);
    topology.push_back(32);
    
    // vytvoření N neuronových sítí
    init_neuron_networks(neuron_networks, topology);
    
    int segment_id = 0;
    int offset = 0;
    int limit = 8;
    
    // otevření db
    data_reader.open();
    
    // získání segmentů z db - pro lepší práci při získávání dat
    data_reader.init_segments(segments);
    
    double min_error = 100.0;
    
    int segment_counter = 0;
    
    int counter = 1;
    
    int target_offset = 0;
    
    // první init parametrů
    segment_id = segments[segment_counter].m_segment_id;
    offset = segments[segment_counter].m_start_id;
    
    int count = 0;
    
    // hlavní tělo programu
    while (true)
    {
        // načtení inputu z DB
        count = data_reader.get_input_data(input_values, limit, offset, segment_id);
        
        // posunu se o zadanej čas o kolik chci predikovat
        target_offset = offset + prediction_for;
        
        // pokud nemám dost hodnot na vstupu
        if (input_values.size() != 8)
        {
            // pokud nejsou už další segmenty
            if (segments.size() < segment_counter + 1)
            {
                break;
            }
            
            // posunu segment (segment představuje skupinu měření)
            segment_counter++;
            
            // nastavím offset + segmentId pro další selecty
            offset = segments[segment_counter].m_start_id;
            segment_id = segments[segment_counter].m_segment_id;
            
            continue;
        }
        
        // načtení predikcí pro vstup
        data_reader.get_prediction_data(prediction_values, limit, target_offset, segment_id);
        
        // pokud nemám k 8 datům 8 předpovědí, jedu dál
        if (prediction_values.size() != 8)
        {
            offset = offset + 8;
            continue;
        }
        
        
//        showVectorVals("Input: ", input_values);
        
  //      showVectorVals("Prediction", prediction_values);
        
        // z predikcí vytvoří vektor pro porovnání výstupu
        prepare_target_values(prediction_values, target_values);
        
        std::mutex m;
        
        // smp
        tbb::parallel_for(size_t(0), neuron_networks.size(), [&](size_t i) {
            /**
            m.lock();
                std::cout << i << std::endl;
            m.unlock();
             */
            neuron_networks[i].feed_forward_propagation(input_values);
            
            neuron_networks[i].back_propagation(target_values);
        });
                          
        
       /*
        std::vector<double> resultValues;
        neuronNetwork.get_results(resultValues);
        
        showVectorVals("Result:", resultValues);
        */
        
        // posuneme offset pro další čtení vstupů
        offset = offset + limit;
        
        // pro debbug, slouží především pro počítání cyklů smyčky
        counter++;
    }
    
    
    for (unsigned i = 0 ; i < neuron_networks.size() ; i++)
    {
        // report jak dobře se síť trénovala, průměr přes všechny vstupy
        std::cout << i << ": Net recent average error: " << neuron_networks[i].get_recent_average_error() << std::endl;
        
        std::cout << i << ": Net error: "
                << neuron_networks[i].get_error() << std::endl;
        
        if (min_error > neuron_networks[i].get_error())
        {
            min_error = neuron_networks[i].get_error();
        }
    }
        
       
    
    
    std::cout << "min error: " << min_error << std::endl;
    
    
    data_reader.close();
    
    std::cout << "Konec" << std::endl;
    
    return 0;
}
