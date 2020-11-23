//
//  main.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//


#include <iostream>
#include "Program_smp.hpp"
#include "Program_gpu.hpp"



void show_open_cl_info()
{
    cl_uint num_devices, i;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);



    cl_device_id* devices = (cl_device_id*)calloc(sizeof(cl_device_id), num_devices);
    
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

    char buf[128];
    for (i = 0; i < num_devices; i++) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, buf, NULL);
        fprintf(stdout, "Device %s supports ", buf);

        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 128, buf, NULL);
        fprintf(stdout, "%s\n", buf);
        
    }

    free(devices);
    
    
    cl_uint numPlatforms = 0;
    cl_int err = clGetPlatformIDs(5, NULL, &numPlatforms);
    
    
   if (CL_SUCCESS == err)
       printf("\nDetected OpenCL platforms: %d ", numPlatforms);
   else
       printf("\nError calling clGetPlatformIDs. Error code: %d", err);
    
    cl_platform_id* platforms = (cl_platform_id*)calloc(sizeof(cl_platform_id), numPlatforms);
    
    for (i = 0; i < numPlatforms; i++) {
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, buf, NULL);
        fprintf(stdout, "Platform %s supports ", buf);

        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 128, buf, NULL);
        fprintf(stdout, "%s\n", buf);
    }
    
    free(platforms);
}

void print_help()
{
    printf("Pouziti\n");
    printf("- Parametry: <predikce> <databaze> <0/1 pro beh na GPU>? <neural.ini>? \n");
    printf("- Parametry s ? jsou volitelne\n");
    printf("- Pro predikci musi platit \"predikce %% 5 = 0\" a je kladna\n");
}

void copy_argument(char*& dest, char* source)
{
    dest = source;
}

int main(int argc, const char * argv[]) {
    
    int prediction;
    char* database;
    char* neural_ini;
    int use_gpu = 0;
    
    if (argc < 3 || argc > 6)
    {
        printf("Chybny pocet parametru\n");
        print_help();
        exit(EXIT_FAILURE);
    }
    
    try
    {
        prediction = std::stoi(argv[1]);
    }
    catch (...)
    {
        printf("Chybna hodnota predikce. Predikce musi byt cele cislo\n");
        print_help();
        exit(EXIT_FAILURE);
    }

    
    // kontrola že je predikce dělitelná 5 a kladne číslo
    if (prediction > 0 && prediction % 5 != 0)
    {
        printf("Chybna hodnota predikce. Predikce musi byt delitelna 5\n");
        print_help();
        exit(EXIT_FAILURE);
    }
    
    // ulozim db
    database = (char *)malloc(strlen(argv[2]) + 1);
    strcpy(database, argv[2]);
    
    
    // kontrola kde ma program bezet
    if (argc > 3)
    {
        try
        {
            int val = std::stoi(argv[3]);
            
            if (val == 0 || val == 1)
            {
                use_gpu = val;
            }
            else
            {
                printf("Chybna hodnota pouziti gpu. Hodnota musí byt 0/1\n");
            }
            
        }
        catch (...)
        {
            printf("Chybna hodnota pouziti gpu. Hodnota musí byt cislo 0/1\n");
            print_help();
            exit(EXIT_FAILURE);
        }
    }
    
    // kontrola ini souboru
    if (argc == 5)
    {
        neural_ini = (char *)malloc(strlen(argv[4]) + 1);
        strcpy(neural_ini, argv[4]);
    }
    
    // podle zadání kde má program běžet spustím program
    if (use_gpu)
    {
        printf("Vybrana platforma GPU\n");
        Program_gpu program_gpu(prediction, database, neural_ini);
        program_gpu.run();
    }
    else
    {
        printf("Vybrana platforma CPU\n");
        Program_smp program_smp(prediction, database, neural_ini);
        program_smp.run(); 
    }
    
    free(neural_ini);
    free(database);
    
    return 0;
}
