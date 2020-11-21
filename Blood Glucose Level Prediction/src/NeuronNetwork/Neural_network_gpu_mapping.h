//
//  Neural_network_gpu_mapping.h
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 21.11.2020.
//

#ifndef Neural_network_gpu_mapping_h
#define Neural_network_gpu_mapping_h

#define NUM_INPUT       8
#define NUM_HIDDEN_1    16
#define NUM_HIDDEN_2    26
#define NUM_OUTPUT      32
#define BIAS_OFFSET     1

namespace Neural_network_gpu_mapping {
    int input_neuron(const int i) { return i; }
    int hidden_neuron_1(const int i) { return 8 + BIAS_OFFSET + i; }
    int hidden_neuron_2(const int i) { return 24 + BIAS_OFFSET + i; }
    int output_neuron(const int i) { return 50 + BIAS_OFFSET + i; }
    int weight_input_hidden(const int i, const int j) { return 100 + i * 16 + j; }
    int weight_hidden_hidden(int i, int j) { return 250 + i * 26 + j; }
    int weight_hidden_output(int i, int j) { return 700 + i * 31 + j; }
    int output_exp_sum() { return 971; }

    // mapování pro trénovací množinu
    int input (int training_set_id, int i) { return training_set_id * NUM_INPUT + i; }

    // mapování pro výsledky
    int expected (int training_set_id, int i) { return training_set_id * NUM_OUTPUT + i; }

    // mapování pro výpočty v backpropagation
    int delta_input_hidden(int i, int j) { return i * 16 + j; }
    int delta_hidden_hidden(int i, int j) { return 150 + i * 26 + j; }
    int delta_hidden_output(int i, int j) { return 600 + i * 31 + j; }
    int error_gradient_hidden_1 (int i) { return 900 + i; }
    int error_gradient_hidden_2 (int i) { return 920 + i; }
    int error_gradien_output (int i) { return 950 + i; }

    // mapování pro výsledky
    int results_index (int entry_id, int i) { return entry_id * NUM_OUTPUT + i; }

}


#endif /* Neural_network_gpu_mapping_h */
