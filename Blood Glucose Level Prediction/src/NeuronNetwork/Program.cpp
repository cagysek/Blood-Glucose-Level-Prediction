//
//  Program.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 09/11/2020.
//

#include "Program.hpp"
#include "Constants.h"

void showVectorVals(const std::string& label, const std::vector<double> &v)
{
    
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << std::ceil(v[i] * 100.0) / 100.0 << " ";
    }

    std::cout << std::endl;
}

Program::Program()
{
    m_data_reader.open("/Users/cagy/Documents/Škola/PPR/Blood-Glucose-Level-Prediction/Blood Glucose Level Prediction/data/asc2018.sqlite");
    
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

Program::~Program()
{
    m_data_reader.close();
}

void Program::run_smp()
{
    // predikce na 60 min
    unsigned prediction_for = 60 / 5; // intervali jsou po 5 min, tímhle zjistím o kolik se posunout

    int segment_id = 0;
    int offset = 0;
    int limit = 8;
    
    double min_error = 100.0;
    
    int segment_counter = 0;
    
    int counter = 1;
    
    int target_offset = 0;
    
    // první init parametrů
    segment_id = m_segments[segment_counter].m_segment_id;
    offset = m_segments[segment_counter].m_start_id;
    
    int count = 0;
    
    auto start = std::chrono::steady_clock::now();
    
    // hlavní tělo programu
    while (true)
    {
        // načtení inputu z DB
        count = m_data_reader.get_input_data(m_input_values, limit, offset, segment_id);
        
        // posunu se o zadanej čas o kolik chci predikovat
        target_offset = offset + prediction_for;
        
        // pokud nemám dost hodnot na vstupu
        if (m_input_values.size() != 8)
        {
            // pokud nejsou už další segmenty
            if (m_segments.size() < segment_counter + 1)
            {
                break;
            }
            
            // posunu segment (segment představuje skupinu měření)
            segment_counter++;
            
            // nastavím offset + segmentId pro další selecty
            offset = m_segments[segment_counter].m_start_id;
            segment_id = m_segments[segment_counter].m_segment_id;
            
            continue;
        }
        
        // načtení predikcí pro vstup
        m_data_reader.get_prediction_data(m_prediction_values, limit, target_offset, segment_id);
        
        // pokud nemám k 8 datům 8 předpovědí, jedu dál
        if (m_prediction_values.size() != 8)
        {
            offset = offset + 8;
            continue;
        }
        
        
//        showVectorVals("Input: ", input_values);
        
  //      showVectorVals("Prediction", prediction_values);
        
        // z predikcí vytvoří vektor pro porovnání výstupu
        prepare_target_values();
        
        std::mutex m;
        
        // smp
        tbb::parallel_for(size_t(0), m_neuron_networks.size(), [&](size_t i) {
            /**
            m.lock();
                std::cout << i << std::endl;
            m.unlock();
             */
            m_neuron_networks[i].feed_forward_propagation(m_input_values);
            
            m_neuron_networks[i].back_propagation(m_target_values);
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
    
    auto end = std::chrono::steady_clock::now();
    
    
    for (unsigned i = 0 ; i < m_neuron_networks.size() ; i++)
    {
        // report jak dobře se síť trénovala, průměr přes všechny vstupy
        std::cout << i << ": Net recent average error: " << m_neuron_networks[i].get_recent_average_error() << std::endl;
        
        std::cout << i << ": Net error: "
                << m_neuron_networks[i].get_error() << std::endl;
        
        if (min_error > m_neuron_networks[i].get_error())
        {
            min_error = m_neuron_networks[i].get_error();
        }
    }
        
    
    std::cout << "min error: " << min_error << std::endl;
    
    std::cout << "Doba výpočtu: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " sec" << std::endl;
    
    std::cout << "Konec" << std::endl;
}

void Program::run_open_cl()
{
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
    int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
    for(i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = i;
    }
 
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("/Users/cagy/Documents/Škola/PPR/Blood-Glucose-Level-Prediction/Blood Glucose Level Prediction/src/NeuronNetwork/neural_network_gpu.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
 
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
            &device_id, &ret_num_devices);
 
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
 
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 
    // Create memory buffers on the device for each vector
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            LIST_SIZE * sizeof(int), NULL, &ret);
 
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
            LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
            LIST_SIZE * sizeof(int), B, 0, NULL, NULL);
 
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&source_str, (const size_t *)&source_size, &ret);
 
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
 
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
 
    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 64; // Divide work items into groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
            &global_item_size, &local_item_size, 0, NULL, NULL);
 
    // Read the memory buffer C on the device to the local variable C
    int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
            LIST_SIZE * sizeof(int), C, 0, NULL, NULL);
 
    // Display the result to the screen
    for(i = 0; i < LIST_SIZE; i++)
        printf("%d + %d = %d\n", A[i], B[i], C[i]);
 
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);

}

void Program::init_neuron_networks(const std::vector<unsigned> &topology)
{
    m_neuron_networks.clear();
    
    unsigned neuron_network_count = 100;
    
    // 100 cca - 13 sec
    // 1000 cca 193 sec
    
    for (unsigned i = 0 ; i < neuron_network_count ; i++)
    {
        m_neuron_networks.push_back(Neuron_network(topology));
    }
}

void Program::prepare_target_values()
{
    // vyčistím targets
    m_target_values.clear();
    
    // vypočitám průměr z předpovědí
    double prediction_average = 0;
    
    for (int i = 0; i < m_prediction_values.size(); i++) {
        prediction_average += m_prediction_values[i];
    }
    
    prediction_average /= m_prediction_values.size();
   
    // vytvořím vektor o velikosti výstupů, které představuje výstup, defaultně dám všude -1 (tanh <-1;1>)
    for (unsigned i = 0 ; i < Constants::Internal_Bound_Count + 2 ; i++)
    {
        m_target_values.push_back(-1);
    }
    
    // z průměrný hodnoty predikce pro hodnoty získám index na výstup (výstup, který by měl mít nejvyšší pravděpodobnost)
    unsigned prediction_index = Constants::Level_To_Index_Band(prediction_average);
    
    // nasadím hodnotu predikovaného výstupu na 1
    m_target_values[prediction_index] = 1;
  
}

