# Linear System Solver using CUDA
Experimenting with CUDA to accelerate a Linear System Solver by leverageing the parallel processing capabilities of a GPU. 

## Linear System and Solution Generator
Status: Done

## LR Factorization (Recursive Algorithm)
Status: Done

## QR Factorization (Iterative Algorithm)
Status: Mostly complete
- Calculation error somewhere in the algorithm

## QR Factorization using CUDA
Status: Mostly complete
- CUDA integration complete
- Same calculation error as the serial version.

## Project Conclusions
The CUDA accelerated QR Factorization program achieves a near 100x speed up to the serial version of the algorithn. 
The LR Program is able to receive nearly the same and in some cases better performance than the CUDA QR program, which leads to the conclusion of how inefficient the implementation I chose to replicate is. 
In the end, I was able to demonstrate the beneifits parallel programming / processing can have on a program, but more works needs to be done to create a parallel program that clearly outperforms the serial algorithms that I implemented. 
