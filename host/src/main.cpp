#include <bitmap.h>
#include <bilateral.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>



#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

void cleanup();

using namespace aocl_utils;
int main() 
{
    double units;
    int num_iters = 1; 
  
    
    //bilateral filter parameters
    uint32_t bilateralRadius = 5;
    float bilateralSigma_squared = .0025;

    char left[] = "left.BMP";//the name of the image file, taken by the arguments
    char right[] = "right.BMP";
    char left_filtered[] = "FPGA_Bilateral_Filter_Left.bmp";
    char right_filtered[] = "FPGA_Bilateral_Filter_Right.bmp";
    
     
   
    //Running Bilateral Filter on FPGA and timing it.
    double start_time_bilateral_fpga = getCurrentTimestamp();
    printf("\nFilter type: Bilateral - FPGA\n");
    for(int i = 0; i < num_iters; ++i) {
      if(b_filter_FPGA(left, right,bilateralRadius,bilateralSigma_squared)==false){
        printf("Failed to call function bilateral_filter\n");
        return -1;
      }
    }
    double elapsed_time_bilateral_fpga = getCurrentTimestamp() - start_time_bilateral_fpga;
    printf("-------------------Bilateral Filter FPGA-------------------\n");
    printf("%d iterations time: %0.3f seconds\n", num_iters, elapsed_time_bilateral_fpga);
    printf("Average single iteration time: %0.3f seconds\n", elapsed_time_bilateral_fpga / num_iters);
    printf("Throughput = %0.3f FPS\n", (float)(num_iters/elapsed_time_bilateral_fpga));
  
     /*
    //Running Bilateral Filter on ARM core and timing it.
    double start_time_bilateral_arm = getCurrentTimestamp();
    printf("\nFilter type: Bilateral - ARM\n");
    for(int i = 0; i < num_iters; ++i) {
      if(b_filter_ARM(left, right,bilateralRadius,bilateralSigma_squared)==false){
        printf("Failed to call function bilateral_filter\n");
        return -1;
      }
    }
    double elapsed_time_bilateral_arm = getCurrentTimestamp() - start_time_bilateral_arm;
    printf("-------------------Bilateral Filter ARM-------------------\n");
    printf("%d iterations time: %0.3f seconds\n", num_iters, elapsed_time_bilateral_arm);
    printf("Average single iteration time: %0.3f seconds\n", elapsed_time_bilateral_arm / num_iters);
    printf("Throughput = %0.3f FPS\n", (float)(num_iters/elapsed_time_bilateral_arm));
   */
    printf("Starting depth\n");
    //depth_mapping_ARM(left_filtered, right_filtered);
    printf("Finished filtering image\n");
    //cleanup(); 
   
    double start_time_bilateral_fpga = getCurrentTimestamp();
    printf("\nFilter type: Depth Extraction - FPGA\n");
    for(int i = 0; i < num_iters; ++i) {
      if(    depth_mapping_FPGA(left_filtered, right_filtered)==false){
        printf("Failed to call function bilateral_filter\n");
        return -1;
      }
    }
    double elapsed_time_bilateral_fpga = getCurrentTimestamp() - start_time_bilateral_fpga;
    printf("-------------------Bilateral Filter FPGA-------------------\n");
    printf("%d iterations time: %0.3f seconds\n", num_iters, elapsed_time_bilateral_fpga);
    printf("Average single iteration time: %0.3f seconds\n", elapsed_time_bilateral_fpga / num_iters);
    printf("Throughput = %0.3f FPS\n", (float)(num_iters/elapsed_time_bilateral_fpga));
  


    return 0;
}

void cleanup()
{
  
  //This is here for aocl utils
}
