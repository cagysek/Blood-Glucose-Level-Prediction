#define NUM_INPUT       8
#define NUM_HIDDEN_1    16
#define NUM_HIDDEN_2    26
#define NUM_OUTPUT      32
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

__kernel void Evaluate1( const int entryIdx, __global float* ND, __global float* TD )
{
    int id = get_global_id( 0 ); // get thread id, 0..INPUTSIZE-1 are valid
    
    if (id >= NUM_INPUT)
    {
        return;
    }

    // nastaví vstupní hodnoty
    ND[input_neuron( id )] = TD[input( id )];

    // defaultní hodnoty pro 1. skrytou vrstvu
    if (id < NUM_HIDDEN_1)
    {
        ND[hidden_neuron_1( id )] = 0;
    }
    
    // defaultní hodnoty pro 1. skrytou vrstvu
    if (id < NUM_HIDDEN_2)
    {
        ND[hidden_neuron_2( id )] = 0;
    }
    // above code must complete before we go into Evaluate2
}

__kernel void Evaluate2( __global float* ND )
{
    int id = get_global_id( 0 ); // get thread id, 0..NUMHIDDEN-1 are valid
    
    if (id >= NUM_HIDDEN_1)
    {
        return;
    }
    // get weighted sum of pattern and bias neuron
    for( int j = 0; j <= NUM_INPUT; j++ )
    {
        ND[hidden_neuron_1( id )] += ND[input_neuron( j )] * ND[weight_input_hidden( id, j )];
    }
    // apply activation function
    ND[hidden_neuron_1( id )] = SigmoidActivationFunction( ND[hidden_neuron_1( id )] );
    
    // above code must complete before we go into Evaluate3
}



