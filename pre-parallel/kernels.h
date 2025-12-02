/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
//__global__ void test_kernel();

 __global__ void ThreeLayerNN(float* W1,float* W2,float* W3,float* b1,float* b2,float* b3,float* train_data,float* train_label,float* losses);
__global__ void VectorMultiplication(float* matrix,float* vector,float* out_vector,int height,int width);
