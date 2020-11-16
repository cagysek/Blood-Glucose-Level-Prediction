#define NUM_INPUT       8
#define NUM_HIDDEN_1    16
#define NUM_HIDDEN_2    26
#define NUM_OUTPUT      32
#define ETA             0.4f
#define ALPHA           0.8f


int input_neuron(int i) { return i; }
int hidden_neuron_1(int i) { return 8 + i; }
int hidden_neuron_2(int i) { return 24 + i; }
int output_neuron(int i) { return 50 + i; }
int weight_input_hidden(int i, int j) { return 100 + i * 16 + j; }
int weight_hidden_hidden(int i, int j) { return 250 + i * 26 + j; }
int weight_hidden_output(int i, int j) { return 700 + i * 31 + j; }
int output_exp_sum() { return 971; }


int input (int i) { return i; }
int expected (int i, int j) { return 8 + i; }


int delta_input_hidden(int i, int j) { return i * 16 + j; }
int delta_hidden_hidden(int i, int j) { return 150 + i * 26 + j; }
int delta_hidden_output(int i, int j) { return 600 + i * 31 + j; }
int error_gradient_hidden_1 (int i) { return 900 + i; }
int error_gradient_hidden_2 (int i) { return 920 + i; }
int error_gradien_output (int i) { return 950 + i; }

/**
    Hello world
 */
__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    C[i] = A[i] + B[i];
}

/**
    aktivační funkce tanH
 */
float tanh_activation_function( float x )
{
    return tanh(x);
}

/**
    Softmax výpočet
 */
float soft_max_function(float x, float exp_sum)
{
    return exp(x) / exp_sum;
}

/**
    Derivace tanH
 */
float tanh_activation_function_derivative( float x )
{
    return 1.0f - (tanh(x) * tanh(x));
}

/**
    Výpočet erroru na výstupu
 */
float get_output_error_gradient( float expected_val, float output_val )
{
    return (expected_val - output_val) * tanh_activation_function_derivative(output_val);
}

/**
    Výpočet error gradientu na 2. skryté vrstvě
 */
float get_hidden_2_error_gradient( int hiddenIdx, __global float* ND, __global float* DG )
{
    float weighted_sum = 0;
    
    // suma hran a error gradientu mezi 2. skrytou a výstupem
    for( int i = 0; i < NUM_OUTPUT; i++ )
    {
        weighted_sum += ND[weight_hidden_output( hiddenIdx, i )] * DG[error_gradien_output( i )];
    }
    
    // suma * derivace aktivační funkce
    return weighted_sum * tanh_activation_function_derivative(ND[hidden_neuron_2( hiddenIdx )]);
}

/**
    Výpočet error gradientu na 1. skryté vrstvě
 */
float get_hidden_1_error_gradient( int hiddenIdx, __global float* ND, __global float* DG )
{
    
    float weighted_sum = 0;
    
    // suma hran a error gradientu mezi skrytými vrstvami
    for( int i = 0; i < NUM_HIDDEN_2; i++ )
    {
        weighted_sum += ND[weight_hidden_hidden( hiddenIdx, i )] * DG[error_gradient_hidden_2( i )];
    }
    
    // suma * derivace aktivační funkce
    return weighted_sum * tanh_activation_function_derivative(ND[hidden_neuron_1( hiddenIdx )]);
}

/**
    Atomická operace pro sčítání
 */
inline void atomic_add_float( volatile __global float* source, const float operand )
{
    union { unsigned int intVal; float floatVal; } newVal;
    union { unsigned int intVal; float floatVal; } prevVal;
    do
    {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    }
    while (atomic_cmpxchg( (volatile __global unsigned int*)source, prevVal.intVal, newVal.intVal ) != prevVal.intVal);
}

/**
    Nasázení vstupních hodnot do neuronů ve všech vrstvách
 */
__kernel void Evaluate1( const int entryIdx, __global float* ND, __global float* TD )
{
    int id = get_global_id( 0 );
    
    if (id >= NUM_OUTPUT)
    {
        return;
    }

    // NUM_HIDDEN_2 -> největší ID co můžu zpracovat
    if (id < NUM_HIDDEN_2)
    {
        // nastaví vstupní hodnoty
        ND[input_neuron( id )] = TD[input( id )];
    }
    
    // defaultní hodnoty pro 1. skrytou vrstvu
    if (id < NUM_HIDDEN_1)
    {
        ND[hidden_neuron_1( id )] = 0;
    }
    
    // defaultní hodnoty pro 2. skrytou vrstvu
    if (id < NUM_HIDDEN_2)
    {
        ND[hidden_neuron_2( id )] = 0;
    }
}

/**
    Feedforward mezi vstupem a 1. skrytou vrstvou
 */
__kernel void feed_forward_input_hidden( __global float* ND )
{
    int id = get_global_id( 0 );
    
    if (id >= NUM_HIDDEN_1)
    {
        return;
    }
    // získání sumy vah a neuronu pro konkrétní neuron
    for( int j = 0; j <= NUM_INPUT; j++ )
    {
        ND[hidden_neuron_1( id )] += ND[input_neuron( j )] * ND[weight_input_hidden( id, j )];
    }
    
    // aplikace aktivační funkce
    ND[hidden_neuron_1( id )] = tanh_activation_function( ND[hidden_neuron_1( id )] );
}

/**
    Feedforward mezi 1. a 2. skrytou vrstvou
 */
__kernel void feed_forward_hidden_hidden( __global float* ND )
{
    int id = get_global_id( 0 );
    
    if (id >= NUM_HIDDEN_2)
    {
        return;
    }
    
    // získání sumy neuronů a vah
    for( int j = 0; j <= NUM_HIDDEN_1; j++ )
    {
        ND[hidden_neuron_2( id )] += ND[hidden_neuron_1( j )] * ND[weight_hidden_hidden( id, j )];
    }
    
    // aplikace aktivační funkce
    ND[hidden_neuron_2( id )] = tanh_activation_function( ND[hidden_neuron_2( id )] );
}

/**
    Feedforward mezi 2. skrytou a výstupní vrstvu
 */
__kernel void feed_forward_hidden_output( __global float* ND )
{
    int id = get_global_id( 0 );
    
    if (id >= NUM_OUTPUT)
    {
        return;
    }
    
    // získání sumy neuronu a vah
    for( int j = 0; j <= NUM_HIDDEN_2; j++ )
    {
        ND[output_neuron( id )] += ND[hidden_neuron_2( j )] * ND[weight_hidden_output( id, j )];
    }
    
    // současně si výsledek v exponencionále uložíme do čítače sumy pro softmax
    // použit atomic add kvůli tomu, že může přistupovat více vláken
    atomic_add_float( &ND[output_exp_sum()],  exp(ND[output_neuron( id )]));
}

/**
    Aplikace softmaxu na výstup, výpočet error gradientu na výstupu
 */
__kernel void backpropagate_1( int entryIdx, __global float* ND, __global float* TD, __global float* DG )
{
    int id = get_global_id( 0 );
    
    // validní idčka
    if (id >= NUM_OUTPUT)
    {
        return;
    }
    
    // aplikace softmaxu na výstupy (na výstup dosud nebyla aktivována žádná aktivační funkce)
    ND[output_neuron( id )] = soft_max_function(ND[output_neuron( id )], ND[output_exp_sum()]);
    
    
    // výpočet error gradientu na výstupech
    DG[error_gradien_output( id )] = get_output_error_gradient( TD[expected( entryIdx, id )], ND[output_neuron( id )] );
}

/**
    Výpočet nových delt mezi výstupem a 2. skrytou vrstvou
 */
__kernel void backpropagate_2( __global float* ND, __global float* DG )
{
    int id = get_global_id( 0 );
    
    // validní idčka
    if (id > NUM_HIDDEN_2)
    {
        return;
    }
    
    // modifikace delt mezi výstupem a 2. skrytou vrstvou
    for( int k = 0; k < NUM_OUTPUT; k++ )
    {
        DG[delta_hidden_output( id, k )] =
                        ETA
                        * ND[hidden_neuron_2( id )]
                        * DG[error_gradien_output( k )]
                        + ALPHA
                        * DG[delta_hidden_output( id, k )];
    }
    
    // výpočet error gradientu pro prvek v 2. skryté vrstvě
    DG[error_gradient_hidden_2( id )] = get_hidden_2_error_gradient( id, ND, DG );
}

/**
    Výpočet nových delt mezi 1. a 2. skrytou vrstvou
 */
__kernel void backpropagate_3( __global float* ND, __global float* DG )
{
    int id = get_global_id( 0 );
    
    // validni idcka
    if (id > NUM_HIDDEN_1)
    {
        return;
    }
    
    // modifikace delt mezi 1. skrytou a 2. skrytou vrstvou
    for( int k = 0; k < NUM_HIDDEN_2; k++ )
    {
        DG[delta_hidden_hidden( id, k )] =
                        ETA
                        * ND[hidden_neuron_1( id )]
                        * DG[error_gradient_hidden_2( k )]
                        + ALPHA
                        * DG[delta_hidden_hidden( id, k )];
    }
    
    // výpočet error gradientu pro prvek v 1. skryté vrstvě
    DG[error_gradient_hidden_1( id )] = get_hidden_1_error_gradient( id, ND, DG );
}

/**
    Výpočet nových delt mezi vstupem a 1. skrytou vrstvou
 */
__kernel void backpropagate_4( __global float* ND, __global float* DG )
{
    int id = get_global_id( 0 );
    
    // validni IDcka
    if (id > NUM_INPUT)
    {
        return;
    }
    
    // modifikace delt mezi vstupem a 1. skrytou vrstvou
    for( int k = 0; k < NUM_HIDDEN_1; k++ )
    {
        DG[delta_input_hidden( id, k )] =
                        ETA
                        * ND[input_neuron( id )]
                        * DG[error_gradient_hidden_1( k )]
                        + ALPHA
                        * DG[delta_input_hidden( id, k )];
    }
}

/**
    update všech vah. Může být více cyklů, navzájem se neovlivní
 */
__kernel void update_weights( __global float* ND, __global float* DG )
{
    int id = get_global_id( 0 );
    
    // 32 je max id co může nastat (32 výstupů)
    if (id >= NUM_OUTPUT)
    {
        return;
    }
    
    // update vah pro input -> 1. skrytá
    if (id <= NUM_INPUT)
    {
        for( int j = 0; j <= NUM_HIDDEN_1; j++ )
        {
            ND[weight_input_hidden( id, j )] += DG[delta_input_hidden( id, j )];
        }
    }
    
    // update vah pro 1. skrytá -> 2. skrytá
    if (id <= NUM_HIDDEN_1)
    {
        for( int j = 0; j <= NUM_HIDDEN_2; j++ )
        {
            ND[weight_hidden_hidden( id, j )] += DG[delta_hidden_hidden( id, j )];
        }
    }
    
    // update vah pro 2. skrytá -> output
    if (id <= NUM_HIDDEN_2)
    {
        for ( int j = 0; j < NUM_OUTPUT; j++ )
        {
            ND[weight_hidden_output( id, j )] += DG[delta_hidden_output( id, j )];
        }
    }
}
