#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DS_timer.cpp"
#define col_size 1024
#define k_size 512
#define row_size 1024
#define divide 32


__global__ void MatMul(float* _a, float* _b, float* _c){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int index = row * blockDim.x* gridDim.x + col;
    _c[index] = 0;
    for(int k = 0; k < k_size; k++){
        _c[index] += _a[row*k_size+k]*_b[col+k*col_size];  
      /*  if(index == 32)
          printf("GPU [%d] is a[%d] %f X b[%d] %f\n",k,row*blockDim.x+k,_a[row*blockDim.x+k],col+k*blockDim.x,_b[col+k*blockDim.x]); */
    }
   // printf("%d is %d\n",index, _c[index]);
}

int main(void){
    float *a, *b, *c, *d;
    float *d_a, *d_b, *d_c;
    int a_size = col_size*k_size;
    int b_size = row_size*k_size;
    dim3 dimBlock(col_size/divide, row_size/divide);
    dim3 dimGrid(divide,divide);

    DS_timer timer(5);
    timer.setTimerName(0, "CUDA Total");
    timer.setTimerName(1, "Computation(Kernel)");
    timer.setTimerName(2, "Data Trans. : Host -> Device");
    timer.setTimerName(3, "Data Trans. : Device -> Host");
    timer.setTimerName(4, "VectorSum on Host");
    timer.initTimers();

    a = new float[a_size];
    memset(a, 0, sizeof(float)*a_size);
    b = new float[b_size];
    memset(b, 0, sizeof(float)*b_size);
    c = new float[row_size*col_size];
    memset(b, 0, sizeof(float)*row_size*col_size);
    d = new float[row_size*col_size];
    memset(b, 0, sizeof(float)*row_size*col_size);

    for(int i = 0; i<a_size;i++){
        a[i] = rand() % 10;
    }
    for(int i = 0; i<b_size;i++){
        b[i] = rand() % 10;
    }

    cudaMalloc(&d_a, sizeof(float)*a_size);
    cudaMalloc(&d_b, sizeof(float)*b_size);
    cudaMalloc(&d_c, sizeof(float)*row_size*col_size);

    timer.onTimer(0);
    timer.onTimer(2);
    cudaMemcpy(d_a, a, sizeof(float)*a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*b_size, cudaMemcpyHostToDevice);
    timer.offTimer(2);
    timer.onTimer(1);
    MatMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize(); // synchronization function
	  timer.offTimer(1);

    timer.onTimer(3);
    cudaMemcpy(c, d_c, sizeof(float)*col_size*row_size, cudaMemcpyDeviceToHost);
	  timer.offTimer(3);
    timer.offTimer(0);
    bool isCorrect = true;

    timer.onTimer(4);
    for(int i = 0; i < row_size; i++){
        for(int j = 0; j < col_size; j++){
            for(int k = 0; k < k_size; k++){
                d[i*col_size+j] += a[i*k_size+k]*b[j+k*col_size];
            /*    if((i*col_size+j)== 32)
                    printf("CPU [%d] is a[%d] %f X b[%d] %f\n",k,i*k_size+k,a[i*k_size+k],j+k*col_size,b[j+k*col_size]); */
            }
        }
    }
    timer.offTimer(4); timer.printTimer();
    for(int i = 0; i < col_size*row_size; i++){
        if(d[i] != c[i]){
            printf("[%d] result is not matched, (%f, %f)\n",i,d[i],c[i]);
            isCorrect = false;
        }
    }

    if(isCorrect){
        printf("Result is same");
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete [] a; delete [] b; delete [] c; delete [] d;
    return 0;
}
