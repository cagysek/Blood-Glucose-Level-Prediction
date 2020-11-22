//
//  Program.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 09/11/2020.
//

#include "Program_smp.hpp"


#define INPUT_COUNT     8
#define OFFSET_CONSTANT 1

void showVectorVals(const std::string& label, const std::vector<double> &v)
{
    
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << std::ceil(v[i] * 100.0) / 100.0 << " ";
    }

    std::cout << std::endl;
}

Program_smp::Program_smp()
{
    m_data_reader.open("/Users/cagy/Documents/Škola/PPR/Blood-Glucose-Level-Prediction/Blood-Glucose-Level-Prediction/data/asc2018.sqlite");
    
    // vytvoření topologie
    std::vector<unsigned> topology;
    topology.push_back(8);
    topology.push_back(16);
    topology.push_back(26);
    topology.push_back(32);
    
    // vytvoření N neuronových sítí
    init_neuron_networks(topology);
    
    // získání segmentů z db - pro lepší práci při získávání dat
    m_data_reader.init_segments(m_segments);
    
}

Program_smp::~Program_smp()
{
    m_data_reader.close();
}

void Program_smp::run()
{
    //return;
    // predikce na 60 min
    unsigned prediction_for = 60 / 5; // intervaly jsou po 5 min, tímhle zjistím o kolik se posunout

    int offset = 0;
    
    double min_error = 100.0;
    
    int counter = 1;
    
    int prediction_index_at = 0;
    
   
    
    // pointer na všechny data z db
    std::vector<Row> *data = m_data_reader.get_data();
    
    auto start = std::chrono::steady_clock::now();
    
    unsigned stop = 40000;
    
    // hlavní tělo programu
  //  while (offset <= data->size() && offset < stop)
    while (offset <= data->size())
    {
       // std::cout << offset << std::endl;
        
        // kolik chci predikovat + počet vstupů (chceme predikci od poslední hodnoty) - 1 (indexujeme od 0)
        prediction_index_at = offset + prediction_for + INPUT_COUNT - 1;
        
        if (prediction_index_at >= data->size())
        {
            break;
        }
        
        
        Row prediction = data->at(prediction_index_at);
        
        // pokud pro hodnotu neexistuje predikce
        // kontrolujeme to s prvním inputem - zajištění, že první hodnota a predikce mají stejný segmentId
        if (prediction.get_segment_id() != data->at(offset).get_segment_id())
        {
            // zvětšíme offset o jedna a pokračujeme
            offset += 1;
            continue;
        }
        
        // inicializace vstupů
        m_input_values.clear();
        
        for (unsigned i = 0; i < INPUT_COUNT ; i++)
        {
            m_input_values.push_back(data->at(offset + i).get_ist());
        }
        
        //showVectorVals("Input: ", m_input_values);
        //std::cout << "Prediction: " << prediction.get_ist() << std::endl;
        
        // z predikcí vytvoří vektor pro porovnání výstupu
        prepare_target_values(prediction.get_ist());
        
        std::mutex m;
        
        // smp
        tbb::parallel_for(size_t(0), m_neuron_networks.size(), [&](size_t i) {
            /**
            m.lock();
                std::cout << i << std::endl;
            m.unlock();
             */
            m_neuron_networks[i].feed_forward_propagation(m_input_values);
            
            m_neuron_networks[i].back_propagation(m_target_values, prediction.get_ist());
        });
                          
        
       /*
        std::vector<double> resultValues;
        neuronNetwork.get_results(resultValues);
        
        showVectorVals("Result:", resultValues);
        */
        
        // posuneme se o jedno
        offset += OFFSET_CONSTANT;
        
        // pro debbug, slouží především pro počítání cyklů smyčky
        counter++;
    }
    
    auto end = std::chrono::steady_clock::now();
    
    Neuron_network best_neuron_network;
    
    for (unsigned i = 0 ; i < m_neuron_networks.size() ; i++)
    {
        // report jak dobře se síť trénovala, průměr přes všechny vstupy
        std::cout << i << ": Net average error: " << m_neuron_networks[i].get_average_error() << std::endl;
        
        std::cout << i << ": Net standart deviation: "
                << m_neuron_networks[i].get_stanadrd_deviation() << std::endl;
        
        double error = m_neuron_networks[i].get_average_error() + m_neuron_networks[i].get_stanadrd_deviation();
        
        if (min_error > error)
        {
            min_error = error;
            best_neuron_network = m_neuron_networks[i];
        }
    }
        
    
    std::cout << "min error: " << min_error << std::endl;
    
    std::cout << "Doba výpočtu: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " sec" << std::endl;
    std::cout << "Počet cyklů: " << counter << std::endl;
    
    std::cout << "Konec" << std::endl;
    
    
    Output_generator output_generator;
    
    output_generator.generate_graph_transmitted_values(best_neuron_network);
    output_generator.generate_graph_transmitted_values_error(best_neuron_network);
    output_generator.generate_init_file(best_neuron_network);
    output_generator.generate_error_csv(best_neuron_network);
    
    free(data);
}


void Program_smp::init_neuron_networks(const std::vector<unsigned> &topology)
{
    m_neuron_networks.clear();
    
    unsigned neuron_network_count = 10;
    
    // 100 cca - 13 sec
    // 1000 cca 193 sec
    
    for (unsigned i = 0 ; i < neuron_network_count ; i++)
    {
        m_neuron_networks.push_back(Neuron_network(topology));
    }
}

void Program_smp::prepare_target_values(double prediction_value)
{
    // vyčistím targets
    m_target_values.clear();
   
    // vytvořím vektor o velikosti výstupů, které představuje výstup, defaultně dám všude 0 (softmax <0;1>)
    for (unsigned i = 0 ; i < 32 ; i++)
    {
        m_target_values.push_back(0);
    }
    
    // z průměrný hodnoty predikce pro hodnoty získám index na výstup (výstup, který by měl mít nejvyšší pravděpodobnost)
    unsigned prediction_index = Constants::Level_To_Index_Band(prediction_value);
    
    // nasadím hodnotu predikovaného výstupu na 1
    m_target_values[prediction_index] = 1;
  
}

