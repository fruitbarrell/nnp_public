/*
    nnp.cu

    Created on: Nov 9, 2025
    Serial implementation of a simple feedforward neural network for MNIST digit classification.

    Network architecture:
    - Input layer: 784 neurons (28x28 pixels)
    - Hidden layer 1: 128 neurons, ReLU activation
    - Hidden layer 2: 64 neurons, ReLU activation
    - Output layer: 10 neurons, Softmax activation

    Training:
    - Loss function: Categorical Cross-Entropy
    - Optimizer: Stochastic Gradient Descent (SGD)
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "config.h"
#include "loader.h"
#include "nnp.h"
#include "kernels.h"


/* Activation functions for relu layers
* Arguments:
*   x: input value
* Returns:
*   activated value based on ReLU function 
*/
// float relu(float x) { return x > 0 ? x : 0; }

// /* Derivative of ReLU activation function
// * Arguments:
// *   y: output value from ReLU function
// * Returns:
// *   derivative value
// */
// float drelu(float y) { return y > 0 ? 1 : 0; }

// /* Softmax activation function
// * Arguments:
// *   z: input array
// *   out: output array to store softmax results
// *   len: length of the input/output arrays
// */ 
void softmax(float *z, float *out, int len) {
    float max = z[0];
    for (int i=1;i<len;i++) if (z[i]>max) max=z[i];
    float sum=0;
    for (int i=0;i<len;i++){ out[i]=expf(z[i]-max); sum+=out[i]; }
    for (int i=0;i<len;i++) out[i]/=sum;
}

/* Initialize weights with small random values
* Arguments:
*   w: weight array to initialize
*   size: number of weights
*/



void init_weights(float *w, int size) {
    for (int i=0;i<size;i++)
        w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
}

/* Train the model using stochastic gradient descent 
* Arguments:
*   model (out): pointer to the MODEL structure which holds network parameters. It is populated by this function.
* Returns:
*   None
*/
void train_model(MODEL* model){
    init_weights(model->W1, SIZE*H1); init_weights(model->b1, H1);
    init_weights(model->W2, H1*H2); init_weights(model->b2, H2);
    init_weights(model->W3, H2*CLASSES); init_weights(model->b3, CLASSES);

    //Create device variables
    float* d_training_data, *d_train_label;
    float *d_W1,*d_W2,*d_W3;
    float *d_b1,*d_b2,*d_b3;

    //allocate and copy training data to GPU

    float* train_data_flat = (float*)malloc(NUM_TRAIN * SIZE * sizeof(float));///
    for (int n = 0; n < NUM_TRAIN; n++)                                         ///This flattens the data to one dim, because
        for (int i = 0; i < SIZE; i++)                                          ///otherwise it gave me problems
            train_data_flat[n * SIZE + i] = train_data[n][i];                 ///

    cudaMalloc(&d_training_data,SIZE*NUM_TRAIN* sizeof(float));
    cudaMemcpy(d_training_data,train_data_flat,SIZE*NUM_TRAIN* sizeof(float),cudaMemcpyHostToDevice);

    float* train_label_flat = (float*)malloc(NUM_TRAIN * CLASSES * sizeof(float));///
    for (int n = 0; n < NUM_TRAIN; n++)                                             ///This flattens the data to one dim, because
        for (int i = 0; i < CLASSES; i++)                                          ///otherwise it gave me problems
            train_label_flat[n * CLASSES + i] = train_label[n][i];               ///

    cudaMalloc(&d_train_label,CLASSES*NUM_TRAIN* sizeof(float));
    cudaMemcpy(d_train_label,train_label_flat,CLASSES*NUM_TRAIN* sizeof(float),cudaMemcpyHostToDevice);

    //Make space for weights in GPU
    cudaMalloc(&d_W1,SIZE*H1* sizeof(float));
    cudaMalloc(&d_W2,H1*H2* sizeof(float));
    cudaMalloc(&d_W3,H2*CLASSES* sizeof(float));
    //Move weights to GPU
    cudaMemcpy(d_W1,model->W1,SIZE*H1* sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2,model->W2,H1*H2* sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3,model->W3,H2*CLASSES* sizeof(float),cudaMemcpyHostToDevice);
    //Same for bias
    cudaMalloc(&d_b1,H1* sizeof(float));
    cudaMalloc(&d_b2,H2* sizeof(float));
    cudaMalloc(&d_b3,CLASSES* sizeof(float));
    
    cudaMemcpy(d_b1,model->b1,H1* sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2,model->b2,H2* sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3,model->b3,CLASSES* sizeof(float),cudaMemcpyHostToDevice);

    // //Create Blocks Constant
    // int blocks=(NUM_TRAIN+BLOCKSIZE-1)/BLOCKSIZE;//this one for the NN part

    //More variables to allocate in the gpu
    float* d_h1,*d_h1a,*d_h2,*d_h2a,*d_out,*d_outa;
    float* d_delta1,*d_delta2,*d_delta3;
    float* d_W1X,*d_W2Z1,*d_W3Z2;

    cudaMalloc(&d_h1,sizeof(float)*H1);   
    cudaMalloc(&d_h1a,sizeof(float)*H1);   
    cudaMalloc(&d_h2,sizeof(float)*H2);
    cudaMalloc(&d_h2a,sizeof(float)*H2);
    cudaMalloc(&d_out,sizeof(float)*CLASSES);   
    cudaMalloc(&d_outa,sizeof(float)*CLASSES);    

    cudaMalloc(&d_delta3,sizeof(float)*CLASSES); 
    cudaMalloc(&d_delta2,sizeof(float)*H2);
    cudaMalloc(&d_delta1,sizeof(float)*H1); 

    cudaMalloc(&d_W1X,sizeof(float)*H1);
    cudaMalloc(&d_W2Z1,sizeof(float)*H2);
    cudaMalloc(&d_W3Z2,sizeof(float)*CLASSES); 

    int H1blocks=(H1+BLOCKSIZE-1)/BLOCKSIZE;
    int H2blocks=(H2+BLOCKSIZE-1)/BLOCKSIZE;
    int CLASSESblocks=(CLASSES+BLOCKSIZE-1)/BLOCKSIZE;
    
    float* d_loss;
    cudaMalloc(&d_loss,sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));
    float h_loss;
    for (int epoch=0; epoch<EPOCHS; epoch++) {
        cudaMemset(d_loss, 0, sizeof(float));
            for(int n=0;n<NUM_TRAIN;n++){
                
                // // ---------- Forward ----------
                vectorMultiply<<<H1blocks,BLOCKSIZE>>>(d_W1,d_training_data+n*SIZE,d_W1X,H1,SIZE);
                reluLayer<<<H1blocks,BLOCKSIZE>>>(d_b1,d_W1X,d_h1a,H1);
                vectorMultiply<<<H2blocks,BLOCKSIZE>>>(d_W2,d_h1a,d_W2Z1,H2,H1);
                reluLayer<<<H2blocks,BLOCKSIZE>>>(d_b2,d_W2Z1,d_h2a,H2);
                vectorMultiply<<<CLASSESblocks,BLOCKSIZE>>>(d_W3,d_h2a,d_W3Z2,CLASSES,H2);
                Layer<<<CLASSESblocks,BLOCKSIZE>>>(d_b3,d_W3Z2,d_out,CLASSES); 
                softmaxGPU<<<1,1>>>(d_out,d_outa,CLASSES);
                // ---------- Loss ----------
                float* d_label_n = d_train_label + n * CLASSES;
                lossGPU<<<1,1>>>(d_label_n,d_outa,d_loss);
                // ---------- Backprop ----------
                delta3<<<CLASSESblocks,BLOCKSIZE>>>(d_label_n,d_outa,d_delta3);
                delta2<<<H2blocks,BLOCKSIZE>>>(d_W3,d_delta3,d_h2a,d_delta2);
                delta1<<<H1blocks,BLOCKSIZE>>>(d_W2,d_delta2,d_h1a,d_delta1);

                // ---------- Update ----------
                //Update W3 and bias 3
                updateWeight<<<H2blocks,BLOCKSIZE>>>(CLASSES,d_W3,d_delta3,d_h2a,H2);
                updateBias<<<CLASSESblocks,BLOCKSIZE>>>(d_b3,d_delta3,CLASSES);

                //Update W2 and bias 2 ERROR BEGINS HERE with delta2 seems like
                updateWeight<<<H1blocks,BLOCKSIZE>>>(H2,d_W2,d_delta2,d_h1a,H1);
                updateBias<<<H2blocks,BLOCKSIZE>>>(d_b2,d_delta2,H2);

                //Update W1 and bias 1
                float* d_sample_n = d_training_data + n * SIZE;
                updateWeight<<<(SIZE+BLOCKSIZE-1)/BLOCKSIZE,BLOCKSIZE>>>(H1,d_W1,d_delta1,d_sample_n,SIZE);
                updateBias<<<H1blocks,BLOCKSIZE>>>(d_b1,d_delta1,H1);

           
        }
        cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        printf("Epoch %d — Loss: %.6f\n", epoch, h_loss);
    }
    // Free training data and labels
    cudaFree(d_training_data);
    cudaFree(d_train_label);

    // Free weights and biases
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_W3);

    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_b3);

    cudaFree(d_h1);
    cudaFree(d_h1a);

    cudaFree(d_h2);
    cudaFree(d_h2a);

    cudaFree(d_out);
    cudaFree(d_outa);

    cudaFree(d_delta3);
    cudaFree(d_delta2);
    cudaFree(d_delta1);

    cudaFree(d_W1X);
    cudaFree(d_W2Z1);
    cudaFree(d_W3Z2);
   

    
}

/* Save the trained model to a binary file
* Arguments:
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None
*/
void save_model(MODEL* model){
	FILE *f = fopen("model.bin", "wb");
	fwrite(model->W1, sizeof(float), SIZE*H1, f);
	fwrite(model->b1, sizeof(float), H1, f);
	fwrite(model->W2, sizeof(float), H1*H2, f);
	fwrite(model->b2, sizeof(float), H2, f);
	fwrite(model->W3, sizeof(float), H2*CLASSES, f);
	fwrite(model->b3, sizeof(float), CLASSES,f);
	fclose(f);
}

/* Load the trained model from a binary file
* Arguments:
*   model (out): pointer to the MODEL structure to populate with loaded weights and biases
* Returns:
*   None
*/
void load_model(MODEL* model){
	FILE *f = fopen("model.bin", "rb");
	fread(model->W1, sizeof(float), SIZE*H1, f);
	fread(model->b1, sizeof(float), H1, f);
	fread(model->W2, sizeof(float), H1*H2, f);
	fread(model->b2, sizeof(float), H2, f);
	fread(model->W3, sizeof(float), H2*CLASSES, f);
	fread(model->b3, sizeof(float), CLASSES, f);
	fclose(f);
}

/* Predict the class of a given input image
* Arguments:
*   x: input image array (flattened 28x28 pixels)
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None (prints predicted class and confidence)
*/
void predict(float *x, MODEL* model){
    float h1[H1], h1a[H1], h2[H2], h2a[H2], out[CLASSES], outa[CLASSES];

    // forward pass
    for (int j=0;j<H1;j++){ 
        h1[j]=model->b1[j]; 
        for(int i=0;i<SIZE;i++) 
            h1[j]+=x[i]*model->W1[i*H1+j]; h1a[j]=relu(h1[j]);
        }
    for (int j=0;j<H2;j++){ 
        h2[j]=model->b2[j]; 
        for(int i=0;i<H1;i++) 
        h2[j]+=h1a[i]*model->W2[i*H2+j]; 
        h2a[j]=relu(h2[j]); 
    }
    for (int k=0;k<CLASSES;k++){ out[k]=model->b3[k]; for(int j=0;j<H2;j++) out[k]+=h2a[j]*model->W3[j*CLASSES+k]; }
    softmax(out,outa,CLASSES);

    // print predicted class
    int pred=0; float max=outa[0];
    for(int k=1;k<CLASSES;k++) if(outa[k]>max){ max=outa[k]; pred=k; }
    printf("Predicted digit: %d (confidence %.2f)\n", pred, max);
}


