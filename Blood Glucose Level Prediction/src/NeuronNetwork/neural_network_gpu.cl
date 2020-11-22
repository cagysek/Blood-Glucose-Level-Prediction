#define NUM_INPUT       8
#define NUM_HIDDEN_1    16
#define NUM_HIDDEN_2    26
#define NUM_OUTPUT      32
#define BIAS_OFFSET     1
#define ETA             0.1f
#define ALPHA           0.8f

// Mapování pro neuronovou síť
int input_neuron(int i) { return i; }
int hidden_neuron_1(int i) { return 8 + BIAS_OFFSET + i; }
int hidden_neuron_2(int i) { return 24 + BIAS_OFFSET + i; }
int output_neuron(int i) { return 50 + BIAS_OFFSET + i; }
int weight_input_hidden(int i, int j) { return 100 + i * 16 + j; }
int weight_hidden_hidden(int i, int j) { return 250 + i * 26 + j; }
int weight_hidden_output(int i, int j) { return 700 + i * 31 + j; }
int output_exp_sum() { return 2000; }

// mapování pro trénovací množinu
int input (int training_set_id, int i) { return training_set_id * NUM_INPUT + i; }

// mapování pro výsledky
int expected (int training_set_id, int i) { return training_set_id * NUM_OUTPUT + i; }

// mapování pro výpočty v backpropagation
int delta_input_hidden(int i, int j) { return i * 16 + j; }
int delta_hidden_hidden(int i, int j) { return 170 + i * 26 + j; }
int delta_hidden_output(int i, int j) { return 700 + i * 32 + j; }
int error_gradient_hidden_1 (int i) { return 1800 + i; }
int error_gradient_hidden_2 (int i) { return 1850 + i; }
int error_gradien_output (int i) { return 1900 + i; }

// mapování pro výsledky
int results_index (int entry_id, int i) { return entry_id * NUM_OUTPUT + i; }


/**
    Hello world
 */
__kernel void vector_add(__global const float *A, __global const int *B, __global int *C) {
 
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
    float weighted_sum = 0.0f;
    
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
    
    float weighted_sum = 0.0f;
    
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
__kernel void neural_network_setup( const int training_set_id, __global float* neural_net, __global float* training_set )
{
    int id = get_global_id( 0 );
    
    if (id >= NUM_OUTPUT)
    {
        return;
    }
    
    neural_net[output_neuron(id)] = 0;
    
    neural_net[output_exp_sum()] = 0.0f;
    
    // NUM_HIDDEN_2 -> největší ID co můžu zpracovat
    if (id < NUM_INPUT)
    {
        // nastaví vstupní hodnoty
        neural_net[input_neuron( id )] = training_set[input( training_set_id, id )];
    }
    
    // defaultní hodnoty pro 1. skrytou vrstvu
    if (id < NUM_HIDDEN_1)
    {
        neural_net[hidden_neuron_1( id )] = 0;
    }
    
    // defaultní hodnoty pro 2. skrytou vrstvu
    if (id < NUM_HIDDEN_2)
    {
        neural_net[hidden_neuron_2( id )] = 0;
    }
}

/**
    Feedforward mezi vstupem a 1. skrytou vrstvou
 */
__kernel void feed_forward_input_hidden( __global float* neural_net )
{
    int id = get_global_id( 0 );
    
    if (id >= NUM_HIDDEN_1)
    {
        return;
    }
    // získání sumy vah a neuronu pro konkrétní neuron
    for( int j = 0; j <= NUM_INPUT; j++ )
    {
  //      printf("1: %f\n", neural_net[input_neuron(j)]);
  //      printf("2: %f\n", neural_net[weight_input_hidden( j, id )]);
        neural_net[hidden_neuron_1( id )] += neural_net[input_neuron( j )] * neural_net[weight_input_hidden( j, id )];
    }

    // aplikace aktivační funkce
    neural_net[hidden_neuron_1( id )] = tanh_activation_function( neural_net[hidden_neuron_1( id )] );
    
    
}

/**
    Feedforward mezi 1. a 2. skrytou vrstvou
 */
__kernel void feed_forward_hidden_hidden( __global float* neural_net )
{
    int id = get_global_id( 0 );
    
    if (id >= NUM_HIDDEN_2)
    {
        return;
    }
    
    // získání sumy neuronů a vah
    for( int j = 0; j <= NUM_HIDDEN_1; j++ )
    {
        neural_net[hidden_neuron_2( id )] += neural_net[hidden_neuron_1( j )] * neural_net[weight_hidden_hidden( j, id )];
    }
    
    // aplikace aktivační funkce
    neural_net[hidden_neuron_2( id )] = tanh_activation_function( neural_net[hidden_neuron_2( id )] );
}

/**
    Feedforward mezi 2. skrytou a výstupní vrstvu
 */
__kernel void feed_forward_hidden_output( __global float* neural_net )
{
    int id = get_global_id( 0 );
    
    if (id >= NUM_OUTPUT)
    {
        return;
    }
    
    // získání sumy neuronu a vah
    for( int j = 0; j <= NUM_HIDDEN_2; j++ )
    {
    //    printf("1: %d %d %f %f\n",j ,id,neural_net[hidden_neuron_2( j )],neural_net[weight_hidden_output( j, id )] );
        
        neural_net[output_neuron( id )] += neural_net[hidden_neuron_2( j )] * neural_net[weight_hidden_output( j, id )];
        
    }
    
    // současně si výsledek v exponencionále uložíme do čítače sumy pro softmax
    // použit atomic add kvůli tomu, že může přistupovat více vláken
    atomic_add_float( &neural_net[output_exp_sum()],  exp(neural_net[output_neuron( id )]));
    
  //  printf("out: %f\n",neural_net[output_exp_sum()] );
}

/**
    Aplikace softmaxu na výstup, výpočet error gradientu na výstupu
 */
__kernel void backpropagate_output( int training_set_id, __global float* neural_net, __global float* target_set, __global float* delta_gradient, __global float* results)
{
    int id = get_global_id( 0 );
    
    // validní idčka
    if (id >= NUM_OUTPUT)
    {
        return;
    }
    
    // aplikace softmaxu na výstupy (na výstup dosud nebyla aktivována žádná aktivační funkce)
 //   printf("prev: %f\n",neural_net[output_neuron( id )] );
 //   printf("sum: %f\n",neural_net[output_exp_sum()] );
    
    neural_net[output_neuron( id )] = soft_max_function(neural_net[output_neuron( id )], neural_net[output_exp_sum()]);
    
 // printf("val: %f\n",neural_net[output_neuron( id )] );
    /*
    neural_net[output_neuron( id )] = tanh_activation_function(neural_net[output_neuron( id )]);
    */
    
    
    // uložím si vypočítané výsledky pro dopočítání errorů na cpu
    // šlo by optimalizovat tady..
    
    results[results_index( training_set_id, id )] = neural_net[output_neuron( id )];
    
 //   printf("g: %f %f %f\n",target_set[expected( training_set_id, id )],neural_net[output_neuron( id )], get_output_error_gradient( target_set[expected( training_set_id, id )], neural_net[output_neuron( id )]));
    
    // výpočet error gradientu na výstupech
    delta_gradient[error_gradien_output( id )] = get_output_error_gradient( target_set[expected( training_set_id, id )], neural_net[output_neuron( id )] );
    
   // printf("%f\n", delta_gradient[error_gradien_output( id )]);
    //printf("%d : %f\n", id, delta_gradient[error_gradien_output( id )]);
//    printf("r %f \n", results[results_index( training_set_id, id )]);
}

/**
    Výpočet nových delt mezi výstupem a 2. skrytou vrstvou
 */
__kernel void backpropagate_output_hidden( __global float* neural_net, __global float* delta_gradient )
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
       // printf("%d, %f, %f, %f\n", k, neural_net[hidden_neuron_2( id )],
       //        delta_gradient[error_gradien_output( k )], delta_gradient[delta_hidden_output( id, k )]);
        delta_gradient[delta_hidden_output( id, k )] =
                        ETA
                        * neural_net[hidden_neuron_2( id )]
                        * delta_gradient[error_gradien_output( k )]
                        + ALPHA
                        * delta_gradient[delta_hidden_output( id, k )];
        
        //printf("%d : %f\n", k, delta_gradient[delta_hidden_output( id, k )]);
    }
    
    // výpočet error gradientu pro prvek v 2. skryté vrstvě
    delta_gradient[error_gradient_hidden_2( id )] = get_hidden_2_error_gradient( id, neural_net, delta_gradient );
}

/**
    Výpočet nových delt mezi 1. a 2. skrytou vrstvou
 */
__kernel void backpropagate_hidden_hidden( __global float* neural_net, __global float* delta_gradient )
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
        delta_gradient[delta_hidden_hidden( id, k )] =
                        ETA
                        * neural_net[hidden_neuron_1( id )]
                        * delta_gradient[error_gradient_hidden_2( k )]
                        + ALPHA
                        * delta_gradient[delta_hidden_hidden( id, k )];
    }
    
    // výpočet error gradientu pro prvek v 1. skryté vrstvě
    delta_gradient[error_gradient_hidden_1( id )] = get_hidden_1_error_gradient( id, neural_net, delta_gradient );
}

/**
    Výpočet nových delt mezi vstupem a 1. skrytou vrstvou
 */
__kernel void backpropagate_hidden_input( __global float* neural_net, __global float* delta_gradient )
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
        delta_gradient[delta_input_hidden( id, k )] =
                        ETA
                        * neural_net[input_neuron( id )]
                        * delta_gradient[error_gradient_hidden_1( k )]
                        + ALPHA
                        * delta_gradient[delta_input_hidden( id, k )];
    }
}

/**
    update všech vah. Může být více cyklů, navzájem se neovlivní
 */
__kernel void update_weights( __global float* neural_net, __global float* delta_gradient )
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
        for( int j = 0; j < NUM_HIDDEN_1; j++ )
        {
         //   printf("%d %d g1: %f\n",id, j, delta_gradient[delta_input_hidden( j, id )]);
            neural_net[weight_input_hidden( j, id )] += delta_gradient[delta_input_hidden( j, id )];
        }
    }
    
    // update vah pro 1. skrytá -> 2. skrytá
    if (id <= NUM_HIDDEN_1)
    {
        for( int j = 0; j < NUM_HIDDEN_2; j++ )
        {
        //    printf("%d %d  G2: %f\n",id,j, delta_gradient[delta_hidden_hidden( j, id )]);
            neural_net[weight_hidden_hidden( j, id )] += delta_gradient[delta_hidden_hidden( j, id )];
        }
    }
    
    // update vah pro 2. skrytá -> output
    if (id <= NUM_HIDDEN_2)
    {
        for ( int j = 0; j < NUM_OUTPUT; j++ )
        {
        //    printf("%d %d  G3: %f\n",id,j,delta_gradient[delta_hidden_output( j, id )]);
            neural_net[weight_hidden_output( j, id )] += delta_gradient[delta_hidden_output( j, id )];
        }
    }
}
