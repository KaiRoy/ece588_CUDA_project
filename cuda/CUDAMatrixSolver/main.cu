#include <iostream>

__global__ void myKernal(void) {

}

int main() {
    myKernal <<<1, 1>>>();
    printf("Hello CUDA!\n");
    return 0;
}