//
//  Program.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 09/11/2020.
//

#include "Program_smp.hpp"


#define INPUT_COUNT     8
#define OFFSET_CONSTANT 1

#define NEURAL_NETWORKS_TO_LEARN 20

void showVectorVals(const std::string& label, const std::vector<double> &v)
{
    
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << std::ceil(v[i] * 100.0) / 100.0 << " ";
    }

    std::cout << std::endl;
}

Program_smp::Program_smp(int prediction, char* database, char* ini_file)
{
    m_prediction = prediction;
    
    printf("Spoustim predpoved hodnot pro %d minut\n", prediction);
    
    m_data_reader.open(database);
    
    // získání segmentů z db - pro lepší práci při získávání dat
    m_data_reader.init_segments(m_segments);
    
    // vytvoření topologie
    std::vector<unsigned> topology;
    topology.push_back(8);
    topology.push_back(16);
    topology.push_back(26);
    topology.push_back(32);
    
    if (ini_file != NULL)
    {
        use_backpropagation = false;
        
        // vytvoření N neuronových sítí
        init_neuron_networks(topology, 1);
        
        // načtení vah z konfiguračního souboru
        load_neuron_network(ini_file);

    }
    else
    {
        init_neuron_networks(topology, NEURAL_NETWORKS_TO_LEARN);
    }
    
}

Program_smp::~Program_smp()
{
    m_data_reader.close();
}

void Program_smp::run()
{
    unsigned prediction_for = m_prediction / 5; // intervaly jsou po 5 min, tímhle zjistím o kolik se posunout

    int counter = 1;
    double min_error = 100.0;
    
    // pointer na všechny data z db
    std::vector<Row> *data = m_data_reader.get_data();
    
    auto start = std::chrono::steady_clock::now();
    
    // hlavní tělo programu
  //  while (offset <= data->size() && offset < stop)
    
        int offset = 0;
        
        
        
        
        
        int prediction_index_at = 0;
    
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
            
            m_neuron_networks[i].feed_forward_propagation(m_input_values, use_backpropagation);
            
            // pokud se jedná pouze
            if (use_backpropagation)
            {
                m_neuron_networks[i].back_propagation(m_target_values, prediction.get_ist());
            }
            else
            {
                m_neuron_networks[i].count_error(m_target_values, prediction.get_ist());
            }
        });
        
        // posuneme se o jedno
        offset += OFFSET_CONSTANT;
        
        // pro debbug, slouží především pro počítání cyklů smyčky
        counter++;
    }
    
    
    auto end = std::chrono::steady_clock::now();
    
    Neuron_network best_neuron_network;
    
    if (use_backpropagation)
    {
        for (unsigned i = 0 ; i < m_neuron_networks.size() ; i++)
        {
            double error = m_neuron_networks[i].get_average_error() + m_neuron_networks[i].get_stanadrd_deviation();
            
            if (min_error > error)
            {
                min_error = error;
                best_neuron_network = m_neuron_networks[i];
            }
        }
    }
    else
    {
        best_neuron_network = m_neuron_networks[0];
        min_error = best_neuron_network.get_average_error() + best_neuron_network.get_stanadrd_deviation();
    }
    
        
    
    std::cout << "Error: " << min_error << std::endl;
    
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


void Program_smp::init_neuron_networks(const std::vector<unsigned> &topology, const unsigned neural_network_to_learn)
{
    m_neuron_networks.clear();
    
    for (unsigned i = 0 ; i < neural_network_to_learn ; i++)
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

/**
    Načtení hodnot z init souboru
 */
void Program_smp::load_neuron_network(char* ini_file)
{
    printf("Nacitam sit ze souboru...\n");
    std::vector<double>* data = m_data_reader.get_ini_data(ini_file);
    int counter = 0;
    
    // pokud se testujeme hodnoty, tak exituje pouze jedna síť
    Neuron_network network = m_neuron_networks[0];
    
    for (unsigned i = 0; i < network.get_layers().size() - 1; i++)
    {
        for (unsigned j = 0; j < network.get_layers()[i].get_neuron_count(); j++)
        {
            for (unsigned k = 0; k < network.get_layers()[i].get_neuron(j).get_weights().size(); k++)
            {
                double val = data->at(counter);
             
                m_neuron_networks[0].get_layer(i).get_neuron(j).get_weight(k).weight = val;
                
                counter++;
            }
        }
    }
    
    free(data);
}

