//
//  File: serial.cpp
//  Author: Kai Roy
//  Description: Serial Program to solve a system of linear equations using LU Factorization
//  References:
//      https://www.baeldung.com/cs/solving-system-linear-equations
//



#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <time.h>
#include <chrono>

#define DEBUG false

using namespace std;

void printMatrix(vector<float> matrix, int size);
void printVector(vector<float> vectorX, int size);

__global__ void matrixTranspose(float* A, int size);
__global__ void matrixCopy(float* B, float*A, int size);
__global__ void matrixMul(float* C, float*A, float* B, int size);
__global__ void vectorMadd(float* A, float* B, float* C, float* s, int size);
__global__ void vectorMul(float* A, float* v, int size);
__global__ void vdiv(float* A, float* B, float* d, int size);
__global__ void mcol(float* A, float* v, int c, int size);
__global__ void vmMul(float* A, float* b, float* c, int size);
__global__ void matrixMinor(float* A, float* B, int d, int size);
__global__ void vnorm(float* A, float* a, int size);
__global__ void vnorm_flip(float* A, float* a, int k, int size);
__global__ void v_e_set(float* e, int k, int size);


void backSub(vector<float> U, vector<float> c, vector<float> x, int size);

int main(int argc, char *argv[])
{
    auto start_time = std::chrono::high_resolution_clock::now();

    string delimiter = " ";
    string s;
    int line_count = 0;
    int size;


    cout << "Serial Matrix Solver Program:" << endl << endl;

    /* ********************** Load Matrix from File ********************** */
    ifstream myfile;
    if (argv[1] != NULL) {
        myfile.open(argv[1]);
    } else {
        myfile.open("backup.txt");
    }

    if (myfile.is_open()) {
        // First Line
        getline(myfile, s);
        size = stoi(s);
        cout << "Matrix size is " << size << "x" << size << endl;
    }

    vector<float> h_matrixA(size * size);
    vector<float> h_vectorB(size);
    vector<float> h_vectorC(size);
    vector<float> h_vectorS(size);

    if (myfile.is_open()) {
        int row = 0;
        while ( getline(myfile, s) ) {
            string temp = s;
            size_t pos = 0;
            string token;
            vector<int> tokens;

            int col = 0;
            while (((pos = temp.find(delimiter)) != string::npos) && (col < size)) {
                token = temp.substr(0, pos);
                tokens.push_back(stoi(token));
                temp.erase(0, pos + delimiter.length());
                col++;
            }
            tokens.push_back(stoi(temp));
            col++;

            for (int i = 0; i < size; i++) {
                // h_matrixA[row][i] = (float)tokens[i];
                h_matrixA[row * size + i] = (float)tokens[i];
            }
            h_vectorB[row] = (float)tokens[size];
            row++;
        }
    }
    myfile.close();

    /* ********************** QR Factorization ********************** */

    size_t mBytes = size * size * sizeof(float);
    size_t vBytes = size * sizeof(float);

    float* d_matrixA;
    float* d_vectorB;
    float* d_vectorC;
    float* d_vectorS;

    cudaMalloc(&d_matrixA, mBytes);
    cudaMalloc(&d_vectorB, vBytes);
    cudaMalloc(&d_vectorC, vBytes);
    cudaMalloc(&d_vectorS, vBytes);

    int THREADS = 32;
    int BLOCKS = (size + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    float* d_Q;
    float* d_R;
    float* d_T;
    float* d_z;
    float* d_z1;
    float* d_q;

    cudaMalloc(&d_Q, mBytes);
    cudaMalloc(&d_R, mBytes);
    cudaMalloc(&d_T, mBytes);
    cudaMalloc(&d_z, mBytes);
    cudaMalloc(&d_z1, mBytes);
    cudaMalloc(&d_q, mBytes);

    float* d_e;
    float* d_x;
    float* d_p;
    cudaMalloc(&d_e, vBytes);
    cudaMalloc(&d_x, vBytes);
    cudaMalloc(&d_p, vBytes);

    float* d_a;
    cudaMalloc(&d_a, sizeof(float));

    vector<float> h_Q(size * size);
    vector<float> h_R(size * size);

    cudaMemcpy(d_matrixA, h_matrixA.data(), mBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectorB, h_vectorB.data(), vBytes, cudaMemcpyHostToDevice);

    matrixCopy<<<blocks, threads>>>(d_z, d_matrixA, size);
	for (int k = 0; k < size-1; k++) {

		matrixMinor<<<blocks, threads>>>(d_z1, d_z, k, size);
        matrixCopy<<<blocks, threads>>>(d_z, d_z1, size);

		mcol<<<blocks, threads>>>(d_z, d_x, k, size);
		vnorm<<<blocks, threads>>>(d_x, d_a, size);
        vnorm_flip<<<blocks, threads>>>(d_matrixA, d_a, k, size);

        v_e_set<<<blocks, threads>>>(d_e, k, size);

		vectorMadd<<<blocks, threads>>>(d_p, d_x, d_e, d_a, size);

        vnorm<<<blocks, threads>>>(d_p, d_a, size);
		vdiv<<<blocks, threads>>>(d_e, d_p, d_a, size);
		vectorMul<<<blocks, threads>>>(d_q, d_e, size);


		matrixMul<<<blocks, threads>>>(d_z1, d_q, d_z, size);
        matrixCopy<<<blocks, threads>>>(d_z, d_z1, size);

        if (k == 0) {
            matrixCopy<<<blocks, threads>>>(d_Q, d_q, size);
            matrixMul<<<blocks, threads>>>(d_R, d_q, d_matrixA, size);
        } else {
            matrixMul<<<blocks, threads>>>(d_z1, d_q, d_Q,  size);
            matrixCopy<<<blocks, threads>>>(d_Q, d_z1, size);
        }
	}

    matrixMul<<<blocks, threads>>>(d_z, d_Q, d_matrixA,  size);
    matrixCopy<<<blocks, threads>>>(d_R, d_z, size);
	// matrixTranspose<<<blocks, threads>>>(d_Q, size);

    vmMul<<<blocks, threads>>>(d_Q, d_vectorB, d_vectorC, size);

    cudaMemcpy(h_Q.data(), d_Q, mBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R.data(), d_R, mBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vectorC.data(), d_vectorC, vBytes, cudaMemcpyDeviceToHost);

    backSub(h_R, h_vectorC, h_vectorS, size);


    clock_t tEnd = clock();

    if (DEBUG) {
        cout << "Constant Vector:" << endl;
        printVector(h_vectorB, size);
        cout << endl;

        cout << "Coefficient Matrix:" << endl;
        printMatrix(h_matrixA, size);
        cout << endl;

        cout << "Q Matrix:" << endl;
        printMatrix(h_Q, size);
        cout  << endl;

        cout << "R Matrix:" << endl;
        printMatrix(h_Q, size);
        cout  << endl;

        cout << "Solution Vector:" << endl;
        printVector(h_vectorS, size);
        cout  << endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;
    std::cout << "Time = " 
    << time/std::chrono::nanoseconds(1) << " nanoseconds\t(" 
    << time/std::chrono::nanoseconds(1)/1000000000 << "." 
    << time/std::chrono::nanoseconds(1)%1000000000 << " sec)\n)";



    cudaFree(d_matrixA);
    // cudaFree(d_matrixB);
    cudaFree(d_Q);
    cudaFree(d_vectorB);
    cudaFree(d_vectorC);
    // cudaFree(d_vectorS);

    return 0;
}

void printMatrix(vector<float> matrix, int size) {
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            cout << matrix[i * size + j] << "\t\t";
        }
        cout << endl;
    }
    cout << endl;
}

void printVector(vector<float> vectorX, int size) {
    for (int i = 0; i < size; i++){
        cout << vectorX[i] << " ";
    }
    cout << endl;
}

__global__ void matrixTranspose(float* A, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row <= col) {
        float t = A[row * size + col];
        A[row * size + col] = A[col * size + row];
        A[col * size + row] = t;
    }
}

__global__ void matrixCopy(float* B, float* A, int size) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    B[row * size + col] = A[row * size + col];

}

__global__ void matrixMul(float* C, float* A, float* B, int size) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    C[row * size + col] = 0;
    for (int k = 0; k < size; k++) {
        C[row * size + col] += A[row * size + k] * B[k * size + col];
    }
}

__global__ void matrixMinor(float* B, float* A, int d, int size) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < d) {
        B[tid * size + tid] = 1;
    }
    if (tid >= d) {
        for (int j = d; j < size; j++)
            B[tid * size + j] = A[tid * size + j];
    }
}


__global__ void vectorMadd(float* C, float* A, float* B, float* s, int size) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (tid < size)
        C[tid] = A[tid] + s[0] * B[tid];
}

__global__ void vectorMul(float* A, float* v, int size) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < size) {
        for (int j = 0; j < size; j++) {
            A[tid * size + j] = -2 *  v[tid] * v[j];
        }
        A[tid * size + tid] += 1;
    }
}  

__global__ void vnorm(float* A, float* a, int size){
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid == 0) {
        float sum = 0;
        for (int i = 0; i < size; i++) 
            sum += A[i] * A[i];

        a[0] = sqrtf(sum);
    }
}

__global__ void vnorm_flip(float* A, float* a, int k, int size){
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid == 0) {
        if (A[k * size + k] > 0)
            a[0] = -a[0];
    }
}
__global__ void v_e_set(float* e, int k, int size){
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < size)
        e[tid] = (tid == k) ? 1 : 0;
}


__global__ void vdiv(float* B, float* A, float* d, int size) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (tid < size)
        B[tid] = A[tid] / d[0];

}

__global__ void mcol(float* A, float* v, int c, int size) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (tid < size)
		v[tid] = A[tid * size + c];
}

__global__ void vmMul(float* A, float* b, float* c, int size) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < size) { 
        float sum = 0;
        for (int j = 0; j < size; j ++) {
            sum += A[tid * size + j] * b[j];
        }
        c[tid] = sum;
    }
}

void backSub(vector<float> U, vector<float> c, vector<float> x, int size) {
    // INPUT
    //    U = upper triangular form
    //    c = vector
    // OUTPUT
    //    the solution vector x

    float sum;
    for (int i = size-1; i >= 0; i--) {
        sum = 0;
        for (int j = i+1; j < size; j ++) {
            sum += x[j] * U[i * size + j];
        }
        x[i] = 1 / U[i * size + i] * (c[i] - sum);
    }
}