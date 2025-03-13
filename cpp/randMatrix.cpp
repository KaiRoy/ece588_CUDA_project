#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#define defaultSize 10
#define restraint 100
#define offset -50
#define DEBUG false

using namespace std;

void printMatrix(int** matrix, int size);
void printSolution(int* solution, int size);

int main(int argc, char *argv[])
{
    string delimiter = " ";
    string s;
    string fileName;
    int line_count = 0;
    int size;
    int** matrix;
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
    matrix = new int*[size];
    for(int i = 0; i < size; ++i)
        matrix[i] = new int[size+1];
    
    // Create Solution Array
    solution = new int[size];

    //Setup RNG
    srand (time(NULL));

    // Generate Solution
    for (int i = 0; i < size; i++) {
        solution[i] = rand() % restraint + offset;
    }

    for (int i = 0; i < size; i++){
        sum = 0;
        for (int j = 0; j < size; j++){
            matrix[i][j] = rand() % restraint + offset;
            sum += matrix[i][j]*solution[j];
        }
        matrix[i][size] = sum;
    }


    /* ********************** Load Matrix to File ********************** */
    ofstream myfile (fileName);
    if (myfile.is_open()) {
        myfile << size << endl;
        for (int i = 0; i < size; i++){
            for (int j = 0; j < size+1; j++){
                if (j == size) {
                    myfile << matrix[i][j];
                } else {
                    myfile << matrix[i][j] << " ";
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
        printMatrix(matrix, size);
    }   

    for (int i = 0; i < size; i++) {
        delete [] matrix[i];
    }
    delete [] matrix;
    delete [] solution;

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
