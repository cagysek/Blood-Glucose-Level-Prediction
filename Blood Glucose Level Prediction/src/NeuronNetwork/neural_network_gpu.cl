#define NUM_INPUT       8
#define NUM_HIDDEN_1    16
#define NUM_HIDDEN_2    32
#define ETA             0.4
#define ALPHA           0.8


int input_neuron(int i) { return i; }
int hidden_neuron_1(int i) { return 8 + i; }
int hidden_neuron_2(int i) { return 24 + i; }
int output_neuron(int i) { return 50 + i; }
int weight_input_hidden(int i, int j) { return 100 + i * 16 + j; }
int weight_hidden_hidden(int i, int j) { return 250 + i * 26 + j; }
int weight_hidden_output(int i, int j) { return 700 + i * 31 + j; }


int input (int i) { return i; }
int expected (int i) { return 8 + i; }


int delta_input_hidden(int i, int j) { return i * 16 + j; }
int delta_hidden_hidden(int i, int j) { return 150 + i * 26 + j; }
int delta_hidden_output(int i, int j) { return 600 + i * 31 + j; }
int error_gradient_hidden_1 (int i) { return 900 + i; }
int error_gradient_hidden_2 (int i) { return 920 + i; }
int error_gradien_output (int i) { return 950 + i; }


__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    C[i] = A[i] + B[i];
}



