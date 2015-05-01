//The Gaussian blur function that runs on the gpu
#define POW2(a) ((a) * (a))
//Focal Length: f=12.5mm 
//Stereo Base 10 mm

__kernel void bilateral_filter(__global const unsigned char *LeftImage, __global const unsigned char *RightImage, const int W, const int H, const int size, const float sigma, __global unsigned char* NewLeft, __global unsigned char* NewRight)
{
    //local variables for iterating through the LeftImage
    //and for holding temporary values
    unsigned int x,y,imgLineSize;
    int i, center,yOff,xOff;
    
    float diff_map_left, gaussian_weight_left,value_left, weight_left, count_left, center_pix_left;
    float diff_map_right, gaussian_weight_right,value_right, weight_right, count_right, center_pix_right;
    
    //find the size of one line of the LeftImage in bytes and the center of the bilateral filter
    imgLineSize = W*3;
    center = size/2;
    
    //Get local index
    i = get_global_id(0);
    
    //Run the window through all of the Left and Right Images
    if(i >= imgLineSize*(size-center)+center*3 && i < W*H*3-imgLineSize*(size-center)-center*3)
    {
        //Resetting values
        count_left       = 0;
        value_left       = 0;
        
        //Obtaining windows center pixel 
        center_pix_left = (float)LeftImage[i+imgLineSize*center + center*3];
        
        //Resetting values
        count_right       = 0;
        value_right       = 0;
        
        //Obtaining windows center pixel 
        center_pix_right = (float)RightImage[i+imgLineSize*center + center*3];
        
        //Vertical window loop
        for(y=0;y<size;y++)
        {
            //Horizontal window loop
            yOff = imgLineSize*(y-center);
            for(x=0;x<size;x++)
            {
                xOff = 3*(x - center);
                
                //Left Image
                diff_map_left = exp (-0.5f *(POW2(center_pix_left - (float)LeftImage[i+xOff+yOff])) * sigma); 
                gaussian_weight_left = exp( - 0.5f * (POW2(x) + POW2(y)) / (size*size));
                
                //printf("diff_map_left value_left %f\n", diff_map_left); 
                weight_left = gaussian_weight_left * diff_map_left;
                value_left += weight_left * LeftImage[i+xOff+yOff];
                count_left += weight_left; 
                
                //Right Image
                diff_map_right = exp (-0.5f *(POW2(center_pix_right - (float)RightImage[i+xOff+yOff])) * sigma); 
                gaussian_weight_right = exp( - 0.5f * (POW2(x) + POW2(y)) / (size*size));
                
                //printf("diff_map_left value_left %f\n", diff_map_left); 
                weight_right = gaussian_weight_right * diff_map_right;
                value_right += weight_right * RightImage[i+xOff+yOff];
                count_right += weight_right; 
            }
        }
        NewLeft[i] = (unsigned char)(value_left / count_left);
        NewRight[i] = (unsigned char)(value_right / count_right);
    }
    else//if it's in the edge keep the same value_left
    {
      NewLeft[i] = LeftImage[i];
      NewRight[i] = RightImage[i];
    }
}
