//
//  Program.hpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 09/11/2020.
//

#ifndef Program_hpp
#define Program_hpp

#include <stdio.h>
#include <iostream>
#include "Neuron_network.hpp"
#include "Data_reader.hpp"

#include "tbb/parallel_for.h"
#include <mutex>

#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.hpp>
#endif

#define MAX_SOURCE_SIZE (0x100000)

class Program
{
public:
    
    Program();
    ~Program();
    void run_smp();
    void run_open_cl();
    
private:
    Data_reader m_data_reader;
    
    // init základních vektorů
    std::vector<Neuron_network> m_neuron_networks;
    std::vector<double> m_input_values;
    std::vector<double> m_target_values;
    std::vector<double> m_prediction_values;
    
    std::vector<Segment> m_segments;
    
    void init_neuron_networks(const std::vector<unsigned> &topology);
    void prepare_target_values(double prediction_value);
};

#endif /* Program_hpp */
