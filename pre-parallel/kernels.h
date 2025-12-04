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
__global__ void vectorMultiply(const float* W, const float* x, float* out, int height, int width);

// Forward-pass kernels
__global__ void reluLayer(float* b, float* WX, float* z_out, int height);
__global__ void Layer(float* b, float* WX, float* z_out, int height);

// Backprop kernels
__global__ void delta3(float* label, float* outa, float* delta3_out);
__global__ void delta2(float* W3, float* delta3, float* h2a, float* delta2_out);
__global__ void delta1(float* W2, float* delta2, float* h1a, float* delta1_out);

// Weight + bias updates
__global__ void updateWeight(int klen, float* W, float* delta, float* layera, int layerlen);
__global__ void updateBias(float* b, float* delta, int layerlen);

__global__ void softmaxGPU(float *z, float *out, int len);
