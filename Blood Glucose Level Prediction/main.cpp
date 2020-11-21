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

int main(int argc, const char * argv[]) {
    
    
    Program_smp program_smp;
    Program_gpu program_gpu;
    

    //program.run_smp();
    
    program_gpu.run();
    
    return 0;
}
