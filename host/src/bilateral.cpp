#include <bilateral.h>
#include <gaussian.h>
#include "CL/opencl.h"
#include <stdint.h>
#include <math.h>
#include "AOCLUtils/aocl_utils.h"


#define MAX_SOURCE_SIZE (1048576) //1 MB
#define MAX_LOG_SIZE    (1048576) //1 MB
#define POW2(a) ((a) * (a))

 //Get platform and device information
  static cl_platform_id platform;
  static cl_device_id device;
  static cl_context context;
  static cl_program program;
  static cl_int status;
  static cl_command_queue queue;
  cl_uint num_devices_b;

using namespace aocl_utils;
// Bilateral filter using the arm core
char b_filter_ARM(char* LeftImage,char* RightImage, uint32_t size, float sigma_squared)
{
    uint32_t i,x,y,imgLineSize; // Used for both because images are same size
    int32_t center,yOff,xOff; // Used for both because images are same size
    float diff_map_left, gaussian_weight_left,value_left, weight_left, count_left, center_pix_left;
    float diff_map_right, gaussian_weight_right,value_right, weight_right, count_right, center_pix_right;

    //read the bitmap
    ME_ImageBMP bmp_left;
    ME_ImageBMP bmp_right;
    if(meImageBMP_Init(&bmp_left,LeftImage)==false)
    {
        printf("Image \"%s\" could not be read as a .BMP file\n",LeftImage);
        return false;
    }
    if(meImageBMP_Init(&bmp_right,RightImage)==false)
    {
        printf("Image \"%s\" could not be read as a .BMP file\n",RightImage);
        return false;
    }
    
    //find the size of one line of the image in bytes and the center of the bilateral filter
    imgLineSize = bmp_left.imgWidth*3;
    center = size/2;
    
    //Run the window through all of the image
    for(i = imgLineSize*(size-center)+center*3; i < (bmp_left.imgHeight*bmp_left.imgWidth*3)-imgLineSize*(size-center)-center*3;i++)
    {   
        count_left       = 0.0f;
        value_left       = 0;
        center_pix_left = (float)bmp_left.imgData[i+imgLineSize*center + center*3];
        
        count_right       = 0.0f;
        value_right       = 0;
        center_pix_right = (float)bmp_right.imgData[i+imgLineSize*center + center*3];

        for(y=0;y<size;y++)
        {
            yOff = imgLineSize*(y-center);
            for(x=0;x<size;x++)
            {
                xOff = 3*(x - center);
                
                //Left Variables
                diff_map_left = exp (-0.5f *(POW2(center_pix_left - (float)bmp_left.imgData[i+xOff+yOff])) * sigma_squared); 
                gaussian_weight_left = exp( - 0.5f * (POW2(x) + POW2(y)) / (size*size));
                weight_left = gaussian_weight_left * diff_map_left;
                value_left += weight_left * bmp_left.imgData[i+xOff+yOff];
                count_left += weight_left; 
                
                //Right Variables
                diff_map_right = exp (-0.5f *(POW2(center_pix_right - (float)bmp_right.imgData[i+xOff+yOff])) * sigma_squared); 
                gaussian_weight_right = exp( - 0.5f * (POW2(x) + POW2(y)) / (size*size));
                weight_right = gaussian_weight_right * diff_map_right;
                value_right += weight_right * bmp_right.imgData[i+xOff+yOff];
                count_right += weight_right; 
            }
        }
        bmp_left.imgData[i] = (unsigned char)(value_left / count_left);
        bmp_right.imgData[i] = (unsigned char)(value_right / count_right);
    }
    //save the image
    char left_name[] = "ARM_Bilateral_Filter_Left.bmp";
    char right_name [] = "ARM_Bilateral_Filter_Right.bmp";
    meImageBMP_Save(&bmp_left,left_name);
    meImageBMP_Save(&bmp_right,right_name);
    return true;
}

//depth_mapping_ARM(ME_ImageBMP* bmp_left, ME_ImageBMP*bmp_right,ME_ImageBMP* bmp_depth)
//Focal Length: f=12.5mm 
//Stereo Base 10 mm
char depth_mapping_ARM(char* LeftImage,char* RightImage)
{
  ME_ImageBMP bmp_left;
  ME_ImageBMP bmp_right;
  ME_ImageBMP bmp_depth; 
  
  if(meImageBMP_Init(&bmp_left,LeftImage)==false)
  {
    printf("Image \"%s\" could not be read as a .BMP file\n",LeftImage);
    return false;
  }
  if(meImageBMP_Init(&bmp_right,RightImage)==false)
  {
    printf("Image \"%s\" could not be read as a .BMP file\n",RightImage);
    return false;
  }
  //Initialize it to get the image formatting
   if(meImageBMP_Init(&bmp_depth,RightImage)==false)
  {
    printf("Image \"%s\" could not be read as a .BMP file\n",RightImage);
    return false;
  }
  
  //Camera Properties
  float focal_length = 12.5; // in mm
  float stereo_base = 10; // in mm
  float focal_stereo_constant = focal_length * stereo_base; 
  
  //Disparity variables
  unsigned char * SAD; // sum of absolute differences  
  float pixel_value, delta_curr; 
  float delta_prev = 0;
  
  //Image variables and loop iterators
  uint32_t i, j, k, imgLineSize,imgSize,imgHeightSize; 
  
  unsigned char *out_image;
  
  //Image parameters
  imgSize = bmp_left.imgWidth*bmp_left.imgHeight*3;
  imgLineSize = bmp_left.imgWidth*3;
  imgHeightSize = bmp_left.imgHeight; 
  
  //Array for new image 
  out_image = (unsigned char *)malloc(imgSize);
  if(out_image == NULL){
    printf("Unable to allocate out_image memory\n");
    return false; 
  }
  //Allocate memory for disparity map
  SAD = (unsigned char *)malloc(imgSize);
  if(SAD == NULL){
    printf("Unable to allocate SAD memory\n");
    return false; 
  }
  
  int window_size = 100; // multiplied by 3 already
  
  //For indexing pixels on the left side of the screen 
  for(i = 0; i < imgHeightSize; i++){
    for(j = 0; j < imgLineSize; j++){
      if(j < imgLineSize-window_size*3){
        pixel_value = (float)bmp_left.imgData[i*imgLineSize+j];
        for(k = 0; k < window_size*3; k++){
          delta_curr = pixel_value - bmp_right.imgData[i*imgLineSize + j + k]; 
          delta_curr = abs(delta_curr); 
          if(delta_prev = 0){
            delta_prev = delta_curr;
            printf("delta_prev is zero\n"); 
          }
          if(delta_prev > delta_curr){
            delta_prev = delta_curr; 
             printf("delta_prev not zero\n"); 
            SAD[i*imgLineSize+j] = (unsigned char)k; // This is the disparity value 
          }
          else
            printf("delta_prev < delta_curr"); 
        }
      }
      else{// In the right corner of left image, do not do anything 
        SAD[i*imgLineSize+j] = 0; 
        printf("Writing 0 to SAD\n");
      }
    }
  }
  printf("After disparity mapping.\n"); 
  for(i = 0; i < imgSize; i++){
    if(SAD[i] != 0)
      out_image[i] = (unsigned char)((focal_stereo_constant) / SAD[i]); 
    else 
      out_image[i] = (unsigned char) 255; 
  }
  
  bmp_depth.imgData = out_image; 
  
    /*
    pixel_value = bmp_left.imgData[i_l]; 
    min_value = 0; 
    curr_value = 0; 
    disparity = 255; 
    */
    
    //indexing has been verified
   /* for(x=0;x<window_size;x++)
    {
      curr_value = abs(pixel_value - bmp_right.imgData[i_r+x]);
      if(min_value > curr_value)
      {
        min_value = curr_value; 
        disparity = x; 
      }
      //else do nothing
     
    }
    */
    //if we are on pixel after(image edge -80) for left image- write 0, else write disparity
    
 
  char ARM_depth[] = "ARM_Depth.bmp";
  meImageBMP_Save(&bmp_depth,ARM_depth);
  
  free(SAD);
  free(out_image);
}

//Bilateral filter the given image using the FPGA
char b_filter_FPGA(char* LeftImage,char* RightImage, uint32_t size,float sigma_squared)
{
    uint32_t imgSize;
    cl_int ret;//the openCL error code/s
    
    //get the image
    ME_ImageBMP bmp_left;
    if(meImageBMP_Init(&bmp_left,LeftImage)==false)
    {
      printf("Image \"%s\" could not be read as a .BMP file\n",LeftImage);
      return false;
    }
    
    ME_ImageBMP bmp_right;
    if(meImageBMP_Init(&bmp_right,RightImage)==false)
    {
      printf("Image \"%s\" could not be read as a .BMP file\n",RightImage);
      return false;
    }
    
    //Same image size for both images assumed
    imgSize = bmp_left.imgWidth*bmp_left.imgHeight*3;
    
    
    //create the pointer that will hold the new (blurred) image data
    unsigned char* newLeft;
    newLeft = (unsigned char *)malloc(imgSize);
    
    unsigned char* newRight;
    newRight = (unsigned char *)malloc(imgSize);
    
    platform = findPlatform("Altera");
    if(platform == NULL) {
      printf("ERROR: Unable to find Altera OpenCL platform.\n");
      return false;
    }
  
    // Query the available OpenCL device.
    cl_device_id *devices = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices_b);
   // printf("Platform: %s\n", getPlatformName(platform).c_str());
   // printf("Found %d device(s)\n", num_devices_b);
    
     // Just use the first device.
    device = devices[0];
    //printf("Using %s\n", getDeviceName(device).c_str());
    delete[] devices;

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Could not create a valid OpenCL context\n");
        return false;
    }
  
    // Create command queue.
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    
    // Create memory buffers on the device for the two images
    cl_mem FPGAImg_left = clCreateBuffer(context,CL_MEM_READ_ONLY,imgSize,NULL,&ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the FPGA image buffer object\n");
        return false;
    }
    
    cl_mem FPGAImg_right = clCreateBuffer(context,CL_MEM_READ_ONLY,imgSize,NULL,&ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the FPGA image buffer object\n");
        return false;
    }
   
    cl_mem FPGANewImg_left = clCreateBuffer(context,CL_MEM_WRITE_ONLY,imgSize,NULL,&ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the FPGA image buffer object\n");
        return false;
    }
    
    cl_mem FPGANewImg_right = clCreateBuffer(context,CL_MEM_WRITE_ONLY,imgSize,NULL,&ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the FPGA image buffer object\n");
        return false;
    }
    
    //Copy the image data kernel to the memory buffer
    if(clEnqueueWriteBuffer(queue, FPGAImg_left, CL_TRUE, 0,imgSize,bmp_left.imgData, 0, NULL, NULL) != CL_SUCCESS)
    {
        printf("Error during sending the image data to the OpenCL buffer, LEFT\n");
        return false;
    }
    if(clEnqueueWriteBuffer(queue, FPGAImg_right, CL_TRUE, 0,imgSize,bmp_right.imgData, 0, NULL, NULL) != CL_SUCCESS)
    {
        printf("Error during sending the image data to the OpenCL buffer, RIGHT\n");
        return false;
    }
 
    
    // Create the program using binary already compiled offline using aoc (i.e. the .aocx file)
    std::string binary_file = getBoardBinaryFile("kernel", device);
    //printf("Using AOCX: %s\n", binary_file.c_str());
    
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
  
    // build the program
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    // Create the OpenCL kernel. This is basically one function of the program declared with the __kernel qualifier
    cl_kernel kernel = clCreateKernel(program, "bilateral_filter", &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Failed to create the OpenCL Kernel from the built program\n");
        return false;
    }
    // Set the arguments of the kernel
    if(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&FPGAImg_left) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"FPGAImg_left\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&FPGAImg_right) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"FPGAImg_left\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 2, sizeof(int), (void *)&bmp_left.imgWidth) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"imageWidth\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 3, sizeof(int), (void *)&bmp_left.imgHeight) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"imgHeight\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel,4,sizeof(int),(void*)&size) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"bilateral size\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel,5,sizeof(int),(void*)&sigma_squared) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"bilateral size\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&FPGANewImg_left) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"FPGANewImg_left\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&FPGANewImg_right) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"FPGANewImg_left\" argument\n");
        return false;
    }
    

    ///enqueue the kernel into the OpenCL device for execution
    size_t globalWorkItemSize = imgSize;//The total size of 1 dimension of the work items. Basically the whole image buffer size
    size_t workGroupSize = 64; //The size of one work group
    
    //Enqueue the actual kernel
    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkItemSize, &workGroupSize, 0, NULL, NULL);


    ///Read the memory buffer of the new image on the device to the new Data local variable
    ret = clEnqueueReadBuffer(queue, FPGANewImg_left, CL_TRUE, 0,imgSize, newLeft, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(queue, FPGANewImg_right, CL_TRUE, 0,imgSize, newRight, 0, NULL, NULL);
    ///Clean up everything
    clFlush(queue);
    clFinish(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(FPGAImg_left);
    clReleaseMemObject(FPGAImg_right);
    clReleaseMemObject(FPGANewImg_left);
    clReleaseMemObject(FPGANewImg_right);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    ///save the new image and return success
    bmp_left.imgData = newLeft;
    bmp_right.imgData = newRight;
    
    
    char left_name[] = "FPGA_Bilateral_Filter_Left.bmp";
    char right_name [] = "FPGA_Bilateral_Filter_Right.bmp";
    meImageBMP_Save(&bmp_left,left_name);
    meImageBMP_Save(&bmp_right,right_name);
  
    free(newLeft);
    free(newRight);
    return true;
}
