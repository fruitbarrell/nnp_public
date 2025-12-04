#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>


const int H1 = 4;
const int H2 = 3;
const int CLASSES = 2;


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

{
    printf("\n=== DELTA TEST ===\n");

    const int H1 = 4;
    const int H2 = 3;
    const int CLASSES = 2;

    float h_label[CLASSES]  = {1, 0};
    float h_out[CLASSES]    = {0.8f, 0.2f};

    float h_h2a[H2] = {1.0f, 2.0f, 3.0f};
    float h_h1a[H1] = {1.5f, -1.0f, 0.5f, 2.0f};

    // small random weights
    float h_W3[H2 * CLASSES] = {
        0.1f, 0.2f,
        -0.3f, 0.4f,
        0.5f, -0.6f
    };

    float h_W2[H1 * H2] = {
        0.2f, -0.1f, 0.3f,
        -0.4f, 0.6f, -0.2f,
        0.1f, 0.2f, -0.5f,
        -0.3f, 0.8f, 0.4f
    };

    float *d_label, *d_out, *d_h2a, *d_h1a, *d_W3, *d_W2;
    float *d_d3, *d_d2, *d_d1;

    cudaMalloc(&d_label, CLASSES * sizeof(float));
    cudaMalloc(&d_out,   CLASSES * sizeof(float));
    cudaMalloc(&d_h2a,   H2 * sizeof(float));
    cudaMalloc(&d_h1a,   H1 * sizeof(float));
    cudaMalloc(&d_W3,    H2 * CLASSES * sizeof(float));
    cudaMalloc(&d_W2,    H1 * H2 * sizeof(float));

    cudaMalloc(&d_d3, CLASSES * sizeof(float));
    cudaMalloc(&d_d2, H2 * sizeof(float));
    cudaMalloc(&d_d1, H1 * sizeof(float));

    cudaMemcpy(d_label, h_label, sizeof(h_label), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out,   h_out,   sizeof(h_out),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_h2a,   h_h2a,   sizeof(h_h2a),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_h1a,   h_h1a,   sizeof(h_h1a),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3,    h_W3,    sizeof(h_W3),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2,    h_W2,    sizeof(h_W2),    cudaMemcpyHostToDevice);

    // ---- RUN KERNELS ----
    delta3<<<1, CLASSES>>>(d_label, d_out, d_d3);
    delta2<<<1, H2>>>(d_W3, d_d3, d_h2a, d_d2);
    delta1<<<1, H1>>>(d_W2, d_d2, d_h1a, d_d1);

    // ---- COPY BACK ----
    float r3[CLASSES], r2[H2], r1[H1];
    cudaMemcpy(r3, d_d3, sizeof(r3), cudaMemcpyDeviceToHost);
    cudaMemcpy(r2, d_d2, sizeof(r2), cudaMemcpyDeviceToHost);
    cudaMemcpy(r1, d_d1, sizeof(r1), cudaMemcpyDeviceToHost);

    // ---- PRINT RESULTS ----
    printf("delta3:\n");
    for (int i = 0; i < CLASSES; i++) printf("%f ", r3[i]);
    printf("\n\ndelta2:\n");
    for (int i = 0; i < H2; i++) printf("%f ", r2[i]);
    printf("\n\ndelta1:\n");
    for (int i = 0; i < H1; i++) printf("%f ", r1[i]);
    printf("\n");

    // ---- CLEAN UP ----
    cudaFree(d_label);
    cudaFree(d_out);
    cudaFree(d_h2a);
    cudaFree(d_h1a);
    cudaFree(d_W3);
    cudaFree(d_W2);
    cudaFree(d_d3);
    cudaFree(d_d2);
    cudaFree(d_d1);

    printf("=== END DELTA TEST ===\n\n");
}