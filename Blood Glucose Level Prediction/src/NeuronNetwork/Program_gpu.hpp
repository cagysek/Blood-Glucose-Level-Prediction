//
//  Program_gpu.hpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 20.11.2020.
//

#ifndef Program_gpu_hpp
#define Program_gpu_hpp

#include <stdio.h>
#include "Data_reader.hpp"
#include "Neuron_network.hpp"



#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.hpp>
#endif

#define MAX_SOURCE_SIZE (0x100000)


class Program_gpu
{
public:
    Program_gpu();
    ~Program_gpu();
    
    void run();
    
    
private:
    Data_reader m_data_reader;
    
    // init základních vektorů
    std::vector<Neuron_network> m_neuron_networks;
    std::vector<cl_float> m_input_values;
    std::vector<double> m_target_values;
    std::vector<double> m_prediction_values_raw;
    
    std::vector<Segment> m_segments;
    
    void init_neuron_networks(const std::vector<unsigned> &topology);
    void prepare_target_values(double prediction_value);
    
    void prepare_training_set();
    void load_kernel_code();
    double get_random();
    
    
    
};

#endif /* Program_gpu_hpp */
