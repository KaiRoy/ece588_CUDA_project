#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <curand.h>

#define defaultSize 10
#define restraint 100
#define offset -50
#define DEBUG false

using namespace std;

void printMatrix(int** matrix, int size);
void printSolution(int* solution, int size);

__global__ void genMatrix(curandGenerator_t generator, int** M, int size);

int main(int argc, char *argv[])
{
    string delimiter = " ";
    string s;
    string fileName;
    int line_count = 0;
    int size;
    int** h_matrix;
    int* solution;
    int sum;

    cout << "Matrix Generator Program" << endl << endl;

    // argc = 1 -> size
    // argc = 2 -> fileName
    if (argc != 3) {
        cout << "Syntax Error: .\\EXE <Matrix Size> <File Name>" << endl;
        return 1;
    }

    if (stoi(argv[1]) >= 0) {
        size = stoi(argv[1]);
        cout << "Matrix will be of size " << size << endl;
    } else {
        cout << "Invalid Size specified, using default size of " << defaultSize << endl;
        size = defaultSize;
    }

    if (argv[2] != NULL) {
        fileName = argv[2];
        cout << "Output file will be named: " << fileName << endl;
    } else {
        cout << "Invalid fileName , using default name default.txt" << endl;
        fileName = "default.txt";
    }

    /* ********************** Generate Random Matrix ********************** */
    // Create Matrix
    h_matrix = new int*[size];
    for(int i = 0; i < size; ++i)
        h_matrix[i] = new int[size+1];
    
    unsigned int** d_matrix;
    size_t bytes = size * size * sizeof(int);
    cudaMalloc(&d_matrix, bytes);

    // Create Solution Array
    solution = new int[size];

    //Setup RNG
    srand (time(NULL));
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, time(NULL));

    int THREADS = 1024;
    int BLOCKS = (size + THREADS - 1) / THREADS;

    // genMatrix<<<BLOCKS, THREADS>>>(generator, d_matrix, size);
    curandGenerate(generator, *d_matrix, size * size);

    cudaMemcpy(h_matrix, d_matrix, bytes, cudaMemcpyDeviceToHost);

    // Generate Solution
    for (int i = 0; i < size; i++) {
        solution[i] = rand() % restraint + offset;
    }

    // for (int i = 0; i < size; i++){
    //     sum = 0;
    //     for (int j = 0; j < size; j++){
    //         h_matrix[i][j] = rand() % restraint + offset;
    //         sum += h_matrix[i][j]*solution[j];
    //     }
    //     h_matrix[i][size] = sum;
    // }
    


    /* ********************** Load Matrix to File ********************** */
    ofstream myfile (fileName);
    if (myfile.is_open()) {
        myfile << size << endl;
        for (int i = 0; i < size; i++){
            for (int j = 0; j < size; j++){
                if (j == size) {
                    myfile << h_matrix[i][j];
                } else {
                    myfile << h_matrix[i][j] << " ";
                }
            }
            if (i != size -1) {
                myfile << endl;
            }
        }
    }
    myfile.close();

    printSolution(solution, size);
    if (DEBUG) {
        printMatrix(h_matrix, size);
    }   

    for (int i = 0; i < size; i++) {
        delete [] h_matrix[i];
    }
    delete [] h_matrix;
    delete [] solution;

    cudaFree(d_matrix);
    curandDestroyGenerator(generator);

    return 0;
}

void printMatrix(int** matrix, int size) {
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void printSolution(int* solution, int size) {
    for (int i = 0; i < size; i++){
        cout << solution[i] << " ";
    }
    cout << endl;
}


__global__ void genMatrix(curandGenerator_t generator, unsigned int** M, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


}