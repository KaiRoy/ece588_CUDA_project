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


int main(int argc, char *argv[])
{
    string delimiter = " ";
    string s;
    string fileName;
    int line_count = 0;
    int size;
    int** matrix;

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
        matrix[i] = new int[size];

    //Setup RNG
    srand (time(NULL));


    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            matrix[i][j] = rand() % restraint + offset;
        }
    }


    /* ********************** Load Matrix to File ********************** */
    ofstream myfile (fileName);
    if (myfile.is_open()) {
        myfile << size << endl;
        for (int i = 0; i < size; i++){
            for (int j = 0; j < size; j++){
                if (j == size -1) {
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

    if (DEBUG) {
        cout << size << endl;
        for (int i = 0; i < size; i++){
            for (int j = 0; j < size; j++){
                cout << matrix[i][j] << "\t";
            }
            cout << endl;
        }
        cout << endl;
    }   

    for (int i = 0; i < size; i++) {
        delete [] matrix[i];
    }
    delete [] matrix;

    return 0;
}
