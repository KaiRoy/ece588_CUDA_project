#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#define DEBUG true

using namespace std;

void printMatrix(int** matrix, int size);

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
    int** matrix;

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

        matrix = new int*[size];
        for(int i = 0; i < size; ++i)
            matrix[i] = new int[size];

        cout << "Matrix Contents:" << endl;
        int row = 0;
        while ( getline(myfile, s) ) {
            string temp = s;
            size_t pos = 0;
            string token;
            vector<int> tokens;

            int col = 0;
            while ((pos = temp.find(delimiter)) != string::npos) {
                token = temp.substr(0, pos);
                tokens.push_back(stoi(token));
                temp.erase(0, pos + delimiter.length());
                col++;
            }
            tokens.push_back(stoi(temp));
            col++;

            for (int i = 0; i < size; i++) {
                matrix[row][i] = tokens[i];
            }
            row++;
        }
    }
    myfile.close();


    /* ********************** Calculate Matrix Solution ********************** */


    if (DEBUG)
        printMatrix(matrix, size);

    // delete matrix
    for (int i = 0; i < size; i++) {
        delete [] matrix[i];
    }
    delete [] matrix;

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
