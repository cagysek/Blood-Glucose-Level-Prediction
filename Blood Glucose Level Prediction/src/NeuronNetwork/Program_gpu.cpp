//
//  Program_gpu.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 20.11.2020.
//

#include "Program_gpu.hpp"
#include "Neural_network_gpu_mapping.h"
#include <random>

#define INPUT_COUNT 8
#define OFFSET_CONSTANT 1

#define NEURAL_NET_SIZE 2048
#define DELTA_GRADIENT_SIZE 2048


Program_gpu::Program_gpu()
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

Program_gpu::~Program_gpu()
{
    m_data_reader.close();
}


void Program_gpu::run()
{
   
    prepare_training_set();
    
    
    
    cl_float* neural_net = (cl_float*)malloc(sizeof(cl_float) * NEURAL_NET_SIZE);
    cl_float* training_set = (cl_float*)malloc(sizeof(cl_float) * m_input_values.size());
    cl_float* target_set = (cl_float*)malloc(sizeof(cl_float) * m_target_values.size());
    cl_float* delta_gradient = (cl_float*)malloc(sizeof(cl_float) * DELTA_GRADIENT_SIZE);
    cl_float* results = (cl_float*)malloc(sizeof(cl_float) * m_target_values.size());
    
    
    
    for(int i = 0; i < m_input_values.size(); i++)
    {
        training_set[i] = m_input_values.at(i);
    }
    
    for (int i = 0; i < m_target_values.size(); i++)
    {
        target_set[i] = m_target_values.at(i);
    }
    
    // nastavení hodnot bias
    neural_net[Neural_network_gpu_mapping::input_neuron(NUM_INPUT)] = 1;
    neural_net[Neural_network_gpu_mapping::hidden_neuron_1(NUM_HIDDEN_1)] = 1;
    neural_net[Neural_network_gpu_mapping::hidden_neuron_2(NUM_HIDDEN_2)] = 1;
    

    // nastavení random vah
    for (int i = 0; i <= NUM_INPUT; i++) {
        for (int j = 0; j < NUM_HIDDEN_1; j++) {
            float rand_val = get_random();
            unsigned index = Neural_network_gpu_mapping::weight_input_hidden(i, j);
            
            neural_net[index] = rand_val;
        }
    }
    
    for (int i = 0; i <= NUM_HIDDEN_1; i++) {
        for (int j = 0; j < NUM_HIDDEN_2; j++) {
            float rand_val = get_random();
            unsigned index = Neural_network_gpu_mapping::weight_hidden_hidden(i, j);
            
            neural_net[index] = rand_val;
        }
    }
    
    for (int i = 0; i <= NUM_HIDDEN_2; i++) {
        for (int j = 0; j < NUM_OUTPUT; j++) {
            float rand_val = get_random();
            unsigned index = Neural_network_gpu_mapping::weight_hidden_output(i, j);
            
            neural_net[index] = rand_val;
        }
    }
 /*
    for (int i = 0; i < 1024; i++) {
        printf("%d: %f \n", i,neural_net[i]);
    }
  */
    
    cl_int ret;
    
    
    
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
    
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1,
            &device_id, &ret_num_devices);
    fprintf(stdout, "Devices %d ", ret_num_devices);
    char buf[128];
    
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, 128, buf, NULL);
    fprintf(stdout, "Device %s supports ", buf);

    clGetDeviceInfo(device_id, CL_DEVICE_VENDOR_ID, 128, buf, NULL);
    fprintf(stdout, "%s\n", buf);
        
    
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
 
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    
    
    
    // Create memory buffers on the device for each vector
    cl_mem neural_net_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
            sizeof(cl_float) * NEURAL_NET_SIZE, NULL, &ret);
    cl_mem training_set_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
            sizeof(cl_float) * m_input_values.size(), NULL, &ret);
    cl_mem target_set_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
            sizeof(cl_float) * m_target_values.size(), NULL, &ret);
    cl_mem delta_gradient_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
            sizeof(cl_float) * DELTA_GRADIENT_SIZE, NULL, &ret);
    cl_mem results_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            sizeof(cl_float) * m_target_values.size(), NULL, &ret);
    
    // zápis hodnot do bufferů
    ret = clEnqueueWriteBuffer(command_queue, neural_net_buffer, CL_TRUE, 0,
           sizeof(cl_float) * NEURAL_NET_SIZE, neural_net, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, training_set_buffer, CL_TRUE, 0,
           sizeof(cl_float) * m_input_values.size(), training_set, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, target_set_buffer, CL_TRUE, 0,
            sizeof(cl_float) * m_target_values.size(), target_set, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, delta_gradient_buffer, CL_TRUE, 0,
            sizeof(cl_float) * DELTA_GRADIENT_SIZE, delta_gradient, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, results_buffer, CL_TRUE, 0,
            sizeof(cl_float) * m_target_values.size(), results, 0, NULL, NULL);
    
     
 
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&source_str, (const size_t *)&source_size, &ret);
 
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
    // Create the OpenCL kernel
    cl_kernel neural_network_setup_kernel = clCreateKernel(program, "neural_network_setup", &ret);
    cl_kernel feed_forward_input_hidden_kernel = clCreateKernel(program, "feed_forward_input_hidden", &ret);
    cl_kernel feed_forward_hidden_hidden_kernel = clCreateKernel(program, "feed_forward_hidden_hidden", &ret);
    cl_kernel feed_forward_hidden_output_kernel = clCreateKernel(program, "feed_forward_hidden_output", &ret);
    cl_kernel backpropagate_output_kernel = clCreateKernel(program, "backpropagate_output", &ret);
    cl_kernel backpropagate_output_hidden_kernel = clCreateKernel(program, "backpropagate_output_hidden", &ret);
    cl_kernel backpropagate_hidden_hidden_kernel = clCreateKernel(program, "backpropagate_hidden_hidden", &ret);
    cl_kernel backpropagate_hidden_input_kernel = clCreateKernel(program, "backpropagate_hidden_input", &ret);
    cl_kernel update_weights_kernel = clCreateKernel(program, "update_weights", &ret);
 
    cl_int training_set_id = 1;
    
    
    //===== Nastavení argumentů pro kernely =======
    //  hodnot neuronove sítě
    clSetKernelArg(neural_network_setup_kernel, 0, sizeof(cl_int), &training_set_id);
    clSetKernelArg(neural_network_setup_kernel, 1, sizeof(cl_mem), &neural_net_buffer);
    clSetKernelArg(neural_network_setup_kernel, 2, sizeof(cl_mem), &training_set_buffer);
    
    // feed forward metody
    clSetKernelArg(feed_forward_input_hidden_kernel, 0, sizeof(cl_mem), &neural_net_buffer);
    clSetKernelArg(feed_forward_hidden_hidden_kernel, 0, sizeof(cl_mem), &neural_net_buffer);
    clSetKernelArg(feed_forward_hidden_output_kernel, 0, sizeof(cl_mem), &neural_net_buffer);
    
    // back propagate metody
    clSetKernelArg(backpropagate_output_kernel, 0, sizeof(cl_int), &training_set_id);
    clSetKernelArg(backpropagate_output_kernel, 1, sizeof(cl_mem), &neural_net_buffer);
    clSetKernelArg(backpropagate_output_kernel, 2, sizeof(cl_mem), &target_set_buffer);
    clSetKernelArg(backpropagate_output_kernel, 3, sizeof(cl_mem), &delta_gradient_buffer);
    clSetKernelArg(backpropagate_output_kernel, 4, sizeof(cl_mem), &results_buffer);
    
    clSetKernelArg(backpropagate_output_hidden_kernel, 0, sizeof(cl_mem), &neural_net_buffer);
    clSetKernelArg(backpropagate_output_hidden_kernel, 1, sizeof(cl_mem), &delta_gradient_buffer);
    
    clSetKernelArg(backpropagate_hidden_hidden_kernel, 0, sizeof(cl_mem), &neural_net_buffer);
    clSetKernelArg(backpropagate_hidden_hidden_kernel, 1, sizeof(cl_mem), &delta_gradient_buffer);
    
    clSetKernelArg(backpropagate_hidden_input_kernel, 0, sizeof(cl_mem), &neural_net_buffer);
    clSetKernelArg(backpropagate_hidden_input_kernel, 1, sizeof(cl_mem), &delta_gradient_buffer);
    
    
    // update vah
    clSetKernelArg(update_weights_kernel, 0, sizeof(cl_mem), &neural_net_buffer);
    clSetKernelArg(update_weights_kernel, 1, sizeof(cl_mem), &delta_gradient_buffer);
    
    size_t input_size = 8;
    size_t hidden_1_size = 16;
    size_t hidden_2_size = 26;
    size_t output_size = 32;
    
    printf("Spouštím výpočet");
    int training_sets_count = m_input_values.size() / input_size;
    for (int i = 0; i < training_sets_count ; i++)
    {
      //  printf("cyklus: %d\n", i);
        // přenastavíme idčka trenovacích množin
        clSetKernelArg(neural_network_setup_kernel, 0, sizeof(cl_int), &i);
        
        clSetKernelArg(backpropagate_output_kernel, 0, sizeof(cl_int), &i);
        
        clEnqueueNDRangeKernel(command_queue, neural_network_setup_kernel, 1, NULL,
                &output_size, NULL, 0, NULL, NULL);
        
        clEnqueueNDRangeKernel(command_queue, feed_forward_input_hidden_kernel, 1, NULL,
                &hidden_1_size, NULL, 0, NULL, NULL);
        
        clEnqueueNDRangeKernel(command_queue, feed_forward_hidden_hidden_kernel, 1, NULL,
                &hidden_2_size, NULL, 0, NULL, NULL);
        
        clEnqueueNDRangeKernel(command_queue, feed_forward_hidden_output_kernel, 1, NULL,
                &output_size, NULL, 0, NULL, NULL);
        
        clEnqueueNDRangeKernel(command_queue, backpropagate_output_kernel, 1, NULL,
                &output_size, NULL, 0, NULL, NULL);
        
        clEnqueueNDRangeKernel(command_queue, backpropagate_output_hidden_kernel, 1, NULL,
                &hidden_2_size, NULL, 0, NULL, NULL);
        
        clEnqueueNDRangeKernel(command_queue, backpropagate_hidden_hidden_kernel, 1, NULL,
                &hidden_1_size, NULL, 0, NULL, NULL);
        
        clEnqueueNDRangeKernel(command_queue, backpropagate_hidden_input_kernel, 1, NULL,
                &input_size, NULL, 0, NULL, NULL);
        
        clEnqueueNDRangeKernel(command_queue, update_weights_kernel, 1, NULL,
                &output_size, NULL, 0, NULL, NULL);
        
        
        ret = clFinish(command_queue);
    }
    printf("Dokončení výpočtu\n");
    

    cl_float *test = (cl_float*)malloc(sizeof(results));
    clEnqueueReadBuffer(command_queue, results_buffer, CL_TRUE, 0, sizeof(results), test, 0, NULL, NULL);
    
    std::vector<float> relative_error_vector;
    
    std::vector<double> aa;
    for (int i = 0; i < training_sets_count; i++) {
        
        unsigned max_index = 0;
        
        // najdu max hodnotu
        for (int j = 0; j < NUM_OUTPUT ; j++)
        {
            float output_val = test[Neural_network_gpu_mapping::results_index(i, j)];
            
            if (output_val > test[Neural_network_gpu_mapping::results_index(i, max_index)])
            {
                max_index = j;
            }
        }
        
        
        // relativní error
        float calculated_prediction = Constants::Band_Index_To_Level(max_index);
        
        float relative_error = abs(calculated_prediction - m_prediction_values_raw.at(i)) / m_prediction_values_raw.at(i);
        
        relative_error_vector.push_back(relative_error);
        
        //std::cout << i << ". " << test[i] << std::endl;
    }
    
    double sum = 0.0;
    
    for (unsigned i = 0; i < relative_error_vector.size() ; i++)
    {
        sum += relative_error_vector[i];
    }
    
    double average_error = sum / relative_error_vector.size();
    
    
    
    std::cout << "Average error: " << average_error << std::endl;
    
    
    double sum2 = 0.0;
    
    for (unsigned i = 0; i < relative_error_vector.size() ; i++)
    {
        sum2 += pow(relative_error_vector[i] - average_error, 2);
    }
    
    double standard_deviation =  sqrt(sum2 / (relative_error_vector.size() - 1));
    
    
    std::cout << "standard deviation: " << standard_deviation << std::endl;
    
    
    // Clean up
    ret = clFlush(command_queue);
    
    
    ret = clReleaseKernel(neural_network_setup_kernel);
    ret = clReleaseKernel(feed_forward_input_hidden_kernel);
    ret = clReleaseKernel(feed_forward_hidden_hidden_kernel);
    ret = clReleaseKernel(feed_forward_hidden_output_kernel);
    ret = clReleaseKernel(backpropagate_output_kernel);
    ret = clReleaseKernel(backpropagate_output_hidden_kernel);
    ret = clReleaseKernel(backpropagate_hidden_hidden_kernel);
    ret = clReleaseKernel(backpropagate_hidden_input_kernel);
    ret = clReleaseKernel(update_weights_kernel);
    
    ret = clReleaseProgram(program);

    ret = clReleaseMemObject(training_set_buffer);
    ret = clReleaseMemObject(neural_net_buffer);
    ret = clReleaseMemObject(target_set_buffer);
    ret = clReleaseMemObject(delta_gradient_buffer);
    ret = clReleaseMemObject(results_buffer);
  
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
 
    free(neural_net);
    free(training_set);
    free(target_set);
    free(results);
    free(delta_gradient);
    free(source_str);
  
}


void Program_gpu::init_neuron_networks(const std::vector<unsigned> &topology)
{
    m_neuron_networks.clear();
    
    unsigned neuron_network_count = 20;
    
    // 100 cca - 13 sec
    // 1000 cca 193 sec
    
    for (unsigned i = 0 ; i < neuron_network_count ; i++)
    {
        m_neuron_networks.push_back(Neuron_network(topology));
    }
}

void Program_gpu::prepare_target_values(double prediction_value)
{   
    // vytvořím vektor o velikosti výstupů, které představuje výstup, defaultně dám všude 0 (softmax <0;1>)
    for (unsigned i = 0 ; i < 32 ; i++)
    {
        m_target_values.push_back(0);
    }
    
    // z průměrný hodnoty predikce pro hodnoty získám index na výstup (výstup, který by měl mít nejvyšší pravděpodobnost)
    unsigned prediction_index = Constants::Level_To_Index_Band(prediction_value);
    
    // nasadím hodnotu predikovaného výstupu na 1
    m_target_values[prediction_index] = 1;
    
    // uložíme si ještě predikovanou hodnotu pro výpočet chyby
    m_prediction_values_raw.push_back(prediction_value);
  
}

void Program_gpu::prepare_training_set()
{
    unsigned prediction_for = 60 / 5; // intervaly jsou po 5 min, tímhle zjistím o kolik se posunout

    int offset = 0;
    
    int counter = 1;
    
    int prediction_index_at = 0;
    
    
    // pointer na všechny data z db
    std::vector<Row> *data = m_data_reader.get_data();
    
    auto start = std::chrono::steady_clock::now();
    
    unsigned stop = 1000;
    
    // inicializace vstupů
    m_input_values.clear();
    m_target_values.clear();
    
    // hlavní tělo programu
  //  while (offset <= data->size() && offset < stop)
    while (offset <= data->size())
    {
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
        
        for (unsigned i = 0; i < INPUT_COUNT ; i++)
        {
            m_input_values.push_back(data->at(offset + i).get_ist());
        }
        
        
        // z predikcí vytvoří vektor pro porovnání výstupu
        prepare_target_values(prediction.get_ist());

        
        // posuneme se o jedno
        offset += OFFSET_CONSTANT;
        
        // pro debbug, slouží především pro počítání cyklů smyčky
        counter++;
    }
    
    auto end = std::chrono::steady_clock::now();
    
}

double Program_gpu::get_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);//uniform distribution between 0 and 1
    
    return dis(gen);
}



