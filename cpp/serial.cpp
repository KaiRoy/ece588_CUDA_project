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

#define DEBUG false

using namespace std;

void printMatrix(float** matrix, int size);
void printVector(float* vec, int size);

void LUdecomp(float** A, float** L, float** U, int size);
void fwdSub(float** L, float* b, float* y, int size);
void backSub(float** U, float* c, float* x, int size);

int main(int argc, char *argv[])
{
    // vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};

    // for (const string& word : msg)
    // {
    //     cout << word << " ";
    // }
    // cout << endl;
    string delimiter = " ";
    string s;
    int line_count = 0;
    int size;
    float** matrixA;
    float** matrixL;
    float** matrixU;
    float* vectorB;
    float* vectorY;
    float* vectorS;


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

        matrixA = new float*[size];
        matrixL = new float*[size];
        matrixU = new float*[size];
        for(int i = 0; i < size; i++) {
            matrixA[i] = new float[size];
            matrixL[i] = new float[size];
            matrixU[i] = new float[size];
        }

        vectorB = new float[size];
        vectorY = new float[size];
        vectorS = new float[size];

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
                matrixA[row][i] = (float)tokens[i];
            }
            vectorB[row] = (float)tokens[size];
            row++;
        }
    }
    myfile.close();


    /* ********************** LU Factorization ********************** */
    LUdecomp(matrixA, matrixL, matrixU, size);
    fwdSub(matrixL, vectorB, vectorY, size);
    backSub(matrixU, vectorY, vectorS, size);


    if (DEBUG) {
        cout << "Resultant Vector:" << endl;
        printVector(vectorB, size);
        cout  << endl;

        cout << "Coefficient Matrix:" << endl;
        printMatrix(matrixA, size);

        cout << "L Matrix:" << endl;
        printMatrix(matrixL, size);
        cout  << endl;
    
        cout << "U Matrix:" << endl;
        printMatrix(matrixU, size);
        cout  << endl;
    }

    cout << "Solution Vector:" << endl;
    printVector(vectorS, size);
    cout  << endl;

    // delete matrix
    for (int i = 0; i < size; i++) {
        delete [] matrixA[i];
        delete [] matrixL[i];
        delete [] matrixU[i];
    }
    delete [] matrixA;
    delete [] matrixL;
    delete [] matrixU;

    delete [] vectorB;
    delete [] vectorY;
    delete [] vectorS;

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


void LUdecomp(float** A, float** L, float** U, int size) {
    // INPUT
    //    A = a non-singular square matrix
    // OUTPUT
    //    Lower unitriangular matrix L 
    //    Upper triangular form U
    int rSize = size - 1;

    // If N = 1, we are done
    if (size == 1) {
        L[0][0] = 1;
        U[0][0] = A[0][0];
        return;
    }

    float a11;
    float l11;
    float u11;

    float* a12 = new float[rSize];
    float* l12 = new float[rSize];
    float* u12 = new float[rSize];

    float* a21 = new float[rSize];
    float* l21 = new float[rSize];
    float* u21 = new float[rSize];


    float** A22 = new float*[rSize];
    float** L22 = new float*[rSize];
    float** U22 = new float*[rSize];
    float** S22 = new float*[rSize];
    for (int i = 0; i < rSize; i++) {
        A22[i] = new float[rSize];
        L22[i] = new float[rSize];
        U22[i] = new float[rSize];
        S22[i] = new float[rSize];
    }

    // Write A as a 2x2 matrix
    a11 = A[0][0];
    for (int i = 0; i < rSize; i++) {
        a12[i] = A[0][i+1];
        a21[i] = A[i+1][0];
        for (int j = 0; j < rSize; j++) {
            A22[i][j] = A[i+1][j+1];
        }
    }

    // Compute first row and column of L, U matrices
    l11 = 1;
    for (int i = 0; i < rSize; i++) {
        l12[i] = 0;
    }

    u11 = a11;
    for (int i = 0; i < rSize; i++) {
        u12[i] = a12[i];
        l21[i] = (1/u11) * a21[i];
    }

    for (int i = 0; i < rSize; i++) {
        for (int j = 0; j < rSize; j++) {
            S22[i][j] = A22[i][j] - (l21[i] * u12[j]);              // Error Here?
        }
    }

    if (DEBUG) {
        cout << "A22 Matrix:" << endl;
        printMatrix(A22, rSize);
        cout  << endl;

        cout << "S22 Matrix:" << endl;
        printMatrix(L22, rSize);
        cout  << endl;
    }

    // Recursively solve subproblem of small size
    LUdecomp(S22, L22, U22, rSize);

    // Put the L, U matrices together
    L[0][0] = l11;
    U[0][0] = u11;
    for (int i = 0; i < rSize; i++) {
        L[0][i+1] = l12[i];
        L[i+1][0] = l21[i];
        U[0][i+1] = u12[i];
        U[i+1][0] = 0;
        for (int j = 0; j < rSize; j++) {
            L[i+1][j+1] = L22[i][j];
            U[i+1][j+1] = U22[i][j];
        }
    }


    delete [] a12;
    delete [] l12;
    delete [] u12;

    delete [] a21;
    delete [] l21;
    delete [] u21;

    for (int i = 0; i < rSize; i++) {
        delete [] A22[i];
        delete [] L22[i];
        delete [] U22[i];
        delete [] S22[i];
    }
    delete [] A22;
    delete [] L22;
    delete [] U22;
    delete [] S22;


    return;
}


void fwdSub(float** L, float* b, float* y, int size) {
    // INPUT
    //    L = Lower triangular form
    //    b = vector b
    // OUTPUT
    //    The solution vector y

    float sum;
    for (int i = 0; i < size; i++) {
        sum = 0;
        for (int j = 0; j < i; j++) {
            sum += y[j] * L[i][j];
        }
        y[i] = (b[i] - sum) / L[i][i];
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