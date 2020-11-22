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
    printf("Spatny vstup\n");
    printf("- Parametry: <predikce> <databaze> <0/1 pro beh na GPU>? <neural.ini>? \n");
    printf("- Parametry s ? jsou volitelne\n");
    printf("- Pro predikci musi platit \"predikce %% 5 = 0\" a je kladna\n");
}


int main(int argc, const char * argv[]) {
    
    int prediction;
    std::string database;
    std::string neural_ini;
    bool use_gpu = false;
    
    if (argc < 3 || argc > 6)
    {
        print_help();
        return 0;
    }
    
    try
    {
        prediction = std::stoi(argv[1]);
    }
    catch (...)
    {
        print_help();
        return 0;
    }

    
    // kontrola že je predikce dělitelná 5 a kladne číslo
    if (prediction > 0 && prediction % 5 != 0)
    {
        print_help();
        return 0;
    }
    
    // ulozim db
    database = std::string(argv[2]);
    
    // kontrola kde ma program bezet
    if (argc > 3)
    {
        try
        {
            use_gpu = std::stoi(argv[3]);
        }
        catch (...)
        {
            print_help();
            return 0;
        }
    }
    
    // kontrola ini souboru
    if (argc == 5)
    {
        neural_ini = std::string(argv[4]);
    }
    
    // podle zadání kde má program běžet spustím program
    if (use_gpu)
    {
        Program_gpu program_gpu;
        program_gpu.run();
    }
    else
    {
        Program_smp program_smp;
        program_smp.run();
    }
    
    
    return 0;
}
