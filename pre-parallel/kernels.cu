#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "config.h"



/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */

__device__ float relu(float x) { return x > 0 ? x : 0; }
__device__ float drelu(float y) { return y > 0 ? 1 : 0; }
__global__ void softmaxGPU(float *z, float *out, int len) {
    float max = z[0];
    for (int i=1;i<len;i++) if (z[i]>max) max=z[i];
    float sum=0;
    for (int i=0;i<len;i++){ out[i]=expf(z[i]-max); sum+=out[i]; }
    for (int i=0;i<len;i++) out[i]/=sum;
}

__global__ void reluLayer(float* b,float* WX,float* z_out,int height){
    int thx=blockIdx.x*blockDim.x+threadIdx.x;
    if (thx>=height) return;
    z_out[thx]= relu(b[thx]+WX[thx]);
}
__global__ void Layer(float* b,float* WX,float* z_out,int height){
    int thx=blockIdx.x*blockDim.x+threadIdx.x;
    if (thx>=height) return;
    z_out[thx]= (b[thx]+WX[thx]);
}




__global__ void delta3(float* label,float* outa,float* delta3_out){
    int thx=blockIdx.x*blockDim.x+threadIdx.x;
    if(thx >= CLASSES) return;
    delta3_out[thx]=label[thx]-outa[thx];
}

__global__ void delta2(float* W3,float* delta3,float* h2a,float* delta2_out){
    int thx=blockIdx.x*blockDim.x+threadIdx.x;
    if(thx >= H2) return;
    float err=0;
    for (int k=0;k<CLASSES;k++) err+=delta3[k]*W3[thx*CLASSES+k];
    delta2_out[thx]=err*drelu(h2a[thx]);
}

__global__ void delta1(float* W2,float* delta2,float* h1a,float* delta1_out){
    int thx=blockIdx.x*blockDim.x+threadIdx.x;
    if(thx >= H1) return;
    float err=0;
    for (int k=0;k<H2;k++) err+=delta2[k]*W2[thx*H2+k];
    delta1_out[thx]=err*drelu(h1a[thx]);
}

__global__ void updateWeight(int klen,float* W,float* delta,float* layera,int layerlen){
    int thx=blockIdx.x*blockDim.x+threadIdx.x;
    if(thx >= layerlen) return;
    for(int k=0;k<klen;k++) W[thx*klen+k]+=LR*delta[k]*layera[thx];
}

__global__ void updateBias(float* b,float*delta,int layerlen){
    int thx=blockIdx.x*blockDim.x+threadIdx.x;
    if(thx >= layerlen) return;
    b[thx]+=LR*delta[thx];
}


__global__ void vectorMultiply(const float* W, const float* x, float* out, int height, int width) {
    int row = blockIdx.x * blockDim.x  + threadIdx.x;;
    if (row >= height) return;

    float sum = 0.0f;
    for (int col = 0; col < width; col++)
        sum += W[row * width + col] * x[col];

    out[row] = sum;
}

__global__ void ThreeLayerNN(float* W1,float* W2,float* W3,float* b1,float* b2,float* b3,float* train_data,float* train_label,float* losses){
   int n=blockIdx.x * blockDim.x + threadIdx.x;
   if (n >= NUM_TRAIN) return;
        }
