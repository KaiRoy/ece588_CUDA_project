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

void printMatrix(float** matrix, int size);
void printVector(float* vec, int size);

void matrixTranspose(float** A, int size);
void matrixCopy(float**A, float** B, int size);
void matrixMul(float**A, float** B, float** C, int size);
void matrixMinor(float** A, float** B, int d, int size);
void vectorMadd(float* A, float* B, float* C, float s, int size);
void vectorMul(float** A, float* v, int size);
float vnorm(float* A, int size);
void vdiv(float* A, float* B, float d, int size);
void mcol(float** A, float* v, int c, int size);
void vmMul(float**A, float* b, float* c, int size);
void backSub(float** U, float* c, float* x, int size);

int main(int argc, char *argv[])
{
    auto start_time = std::chrono::high_resolution_clock::now();

    string delimiter = " ";
    string s;
    int line_count = 0;
    int size;
    float** h_matrixA;
    float* h_vectorB;
    float* h_vectorS;


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

        h_matrixA = new float*[size];
        for(int i = 0; i < size; i++) {
            h_matrixA[i] = new float[size];
        }

        h_vectorB = new float[size];
        h_vectorS = new float[size];

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
                h_matrixA[row][i] = (float)tokens[i];
            }
            h_vectorB[row] = (float)tokens[size];
            row++;
        }
    }
    myfile.close();


    /* ********************** QR Factorization ********************** */

    float** Q = new float*[size];
    float** R = new float*[size];
    // float** T = new float*[size];
    float*** q = new float**[size];

    float** z = new float*[size];
    float** z1 = new float*[size];
    for(int i = 0; i < size; i++) {
        Q[i] = new float[size];
        R[i] = new float[size];
        // T[i] = new float[size];
        z[i] = new float[size];
        z1[i] = new float[size];

        q[i] = new float*[size];
        for(int j = 0; j < size; j++) {
            q[i][j] = new float[size];
        }
    }

    matrixCopy(h_matrixA, z, size);
	for (int k = 0; k < size-1; k++) {
		float* e = new float[size];
        float* x = new float[size];
        float* p = new float[size];
        float a;

		matrixMinor(z, z1, k, size);
        matrixCopy(z1, z, size);

		mcol(z, x, k, size);
		a = vnorm(x, size);
		if (h_matrixA[k][k] > 0) a = -a;

		for (int i = 0; i < size; i++)
			e[i] = (i == k) ? 1 : 0;

		vectorMadd(x, e, p, a, size);
		vdiv(p, e, vnorm(p, size), size);
		vectorMul(q[k], e, size);


		matrixMul(q[k], z, z1, size);
        matrixCopy(z1, z, size);
        delete [] e;
        delete [] x;
	}

    matrixCopy(q[0], Q, size);
	matrixMul(q[0], h_matrixA, R, size);
	for (int i = 1; i < size-1; i++) {
		matrixMul(q[i], Q, z1, size);
        matrixCopy(z1, Q, size);
	}

    matrixMul(Q, h_matrixA, z, size);
    matrixCopy(z, R, size);
	matrixTranspose(Q, size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            delete [] q[i][j];
        }
        delete [] q[i];
    }
    delete [] q;

    for (int i = 0; i < size; i++) {
        delete [] z[i];
        delete [] z1[i];
    }
    delete [] z;
    delete [] z1;



    backSub(R, h_vectorB, h_vectorS, size);

    clock_t tEnd = clock();

    if (DEBUG) {
        // cout << "Resultant Vector:" << endl;
        // printVector(h_vectorB, size);
        // cout  << endl;

        cout << "Q Matrix:" << endl;
        printMatrix(Q, size);
    
        cout << "R Matrix:" << endl;
        printMatrix(R, size);
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

    // delete matrix
    for (int i = 0; i < size; i++) {
        delete [] h_matrixA[i];
        delete [] R[i];
    }
    delete [] h_matrixA;
    delete [] R;

    delete [] h_vectorB;
    delete [] h_vectorS;



    return 0;
}

void printMatrix(float** matrix, int size) {
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            cout << matrix[i][j] << "\t\t";
        }
        cout << endl;
    }
    cout << endl;
}

void printVector(float* vectorX, int size) {
    for (int i = 0; i < size; i++){
        cout << vectorX[i] << " ";
    }
    cout << endl;
}

void matrixTranspose(float** A, int size) {

    for(int i = 0; i < size; i++) {
        for (int j = 0; j < i; j++) {
            float t = A[i][j];
			A[i][j] = A[j][i];
			A[j][i] = t;
        }
    }

    return;
}

void matrixCopy(float**A, float** B, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            B[i][j] = A[i][j];
        }
    }
}

void matrixMul(float**A, float** B, float** C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrixMinor(float** A, float** B, int d, int size) {
    for (int i = 0; i < d; i++)
        B[i][i] = 1;
    for (int i = d; i < size; i++)
        for (int j = d; j < size; j++)
            B[i][j] = A[i][j];

    return;
}

void vectorMadd(float* A, float* B, float* C, float s, int size) {
    for (int i = 0; i < size; i++)
        C[i] = A[i] + s * B[i];
}

void vectorMul(float** A, float* v, int size) {
    for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			A[i][j] = -2 *  v[i] * v[j];
	for (int i = 0; i < size; i++)
		A[i][i] += 1;
}  

float vnorm(float* A, int size)
{
	float sum = 0;
	for (int i = 0; i < size; i++) 
        sum += A[i] * A[i];

	return sqrt(sum);
}

void vdiv(float* A, float* B, float d, int size)
{
	for (int i = 0; i < size; i++) 
        B[i] = A[i] / d;

	return;
}

void mcol(float** A, float* v, int c, int size)
{
	for (int i = 0; i < size; i++)
		v[i] = A[i][c];

	return;
}

void vmMul(float**A, float* b, float* c, int size) {
    for(int i = 0; i < size; i++) {
        float sum = 0;
        for (int j = 0; j < size; j ++) {
            sum += A[i][j] * b[j];
        }
        c[i] = sum;
    }
}

void backSub(float** U, float* c, float* x, int size) {
    // INPUT
    //    U = upper triangular form
    //    c = vector
    // OUTPUT
    //    the solution vector x

    float sum;
    for (int i = size-1; i >= 0; i--) {
        sum = 0;
        for (int j = i+1; j < size; j ++) {
            sum += x[j] * U[i][j];
        }
        x[i] = 1 / U[i][i] * (c[i] - sum);
    }
}