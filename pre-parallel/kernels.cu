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


__global__ void softmax(float *z, float *out, int len) {
    float max = z[0];
    for (int i=1;i<len;i++) if (z[i]>max) max=z[i];
    float sum=0;
    for (int i=0;i<len;i++){ out[i]=expf(z[i]-max); sum+=out[i]; }
    for (int i=0;i<len;i++) out[i]/=sum;
}
  // // ---------- Backprop ----------
            // float delta3[CLASSES];
            // for (int k=0;k<CLASSES;k++)
            //     delta3[k] = train_label[n*CLASSES+k]-outa[k];

            // float delta2[H2];
            // for (int j=0;j<H2;j++){
            //     float err=0;
            //     for (int k=0;k<CLASSES;k++) err+=delta3[k]*W3[j*CLASSES+k];
            //     delta2[j]=err*drelu(h2a[j]);
            // }

            // float delta1[H1];
            // for (int j=0;j<H1;j++){
            //     float err=0;
            //     for (int k=0;k<H2;k++) err+=delta2[k]*W2[j*H2+k];
            //     delta1[j]=err*drelu(h1a[j]);
            // }
__global__ void delta3(float* label,float* outa,float* delta3_out){
    int thx=blockIdx.x*blockDim.x+threadIdx.x;
    if(thx >= CLASSES) return;
    delta3_out[thx]=label[thx]-outa[thx];
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
//    int n=blockIdx.x * blockDim.x + threadIdx.x;
//    if (n >= NUM_TRAIN) return;
//     // ---------- Forward ----------
//             float h1[H1], h1a[H1];
//             for (int j=0;j<H1;j++){
//                 h1[j]=b1[j];
//                 for (int i=0;i<SIZE;i++) 
//                     h1[j]+=train_data[n*SIZE+i]*W1[i*H1+j];
//                 h1a[j]=relu(h1[j]);
//             }
//             float h2[H2], h2a[H2];
//             for (int j=0;j<H2;j++){
//                 h2[j]=b2[j];
//                 for (int i=0;i<H1;i++) 
//                     h2[j]+=h1a[i]*W2[i*H2+j];
//                 h2a[j]=relu(h2[j]);
//             }
//             float out[CLASSES], outa[CLASSES];
//             for (int k=0;k<CLASSES;k++){
//                 out[k]=b3[k];
//                 for (int j=0;j<H2;j++) out[k]+=h2a[j]*W3[j*CLASSES+k];
//             }
//             softmax(out,outa,CLASSES);
           

//             // ---------- Backprop ----------
//             float delta3[CLASSES];
//             for (int k=0;k<CLASSES;k++)
//                 delta3[k] = train_label[n*CLASSES+k]-outa[k];

//             float delta2[H2];
//             for (int j=0;j<H2;j++){
//                 float err=0;
//                 for (int k=0;k<CLASSES;k++) err+=delta3[k]*W3[j*CLASSES+k];
//                 delta2[j]=err*drelu(h2a[j]);
//             }

//             float delta1[H1];
//             for (int j=0;j<H1;j++){
//                 float err=0;
//                 for (int k=0;k<H2;k++) err+=delta2[k]*W2[j*H2+k];
//                 delta1[j]=err*drelu(h1a[j]);
//             }

//             // ---------- Update ----------
//             for (int j=0;j<H2;j++)
//                 for (int k=0;k<CLASSES;k++)
//                     // W3[j*CLASSES+k]+=LR*delta3[k]*h2a[j];
//                     atomicAdd(&W3[j*CLASSES+k],LR*delta3[k]*h2a[j]);
//             for (int k=0;k<CLASSES;k++) /*b3[k]+=LR*delta3[k];*/ atomicAdd(&b3[k],LR*delta3[k]);

//             for (int j=0;j<H1;j++)
//                 for (int k=0;k<H2;k++)
//                     // W2[j*H2+k]+=LR*delta2[k]*h1a[j];
//                     atomicAdd(&W2[j*H2+k],LR*delta2[k]*h1a[j]);
//             for (int k=0;k<H2;k++) /*b2[k]+=LR*delta2[k];*/ atomicAdd(&b2[k],LR*delta2[k]);

//             for (int i=0;i<SIZE;i++)
//                 for (int j=0;j<H1;j++)
//                     // W1[i*H1+j]+=LR*delta1[j]*train_data[n*SIZE+i];
//                     atomicAdd(&W1[i*H1+j],LR*delta1[j]*train_data[n*SIZE+i]);
//             for (int j=0;j<H1;j++) /*b1[j]+=LR*delta1[j];*/  atomicAdd(&b1[j],LR*delta1[j]);

//             // ---------- Loss ----------
//             float loss = 0.0f;
//             for (int k=0;k<CLASSES;k++)
//                 loss -= train_label[n*CLASSES+k]*logf(outa[k]+1e-8f);
//             losses[n]=loss;
        }
