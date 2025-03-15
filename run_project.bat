CLS
CLS
@echo Data Colelction Start
@echo -----------------------------------------
@echo ""

SET /A x = 10 
.\randMatrix.exe %x% test.txt
.\parallel.exe test.txt
@echo ------------------ %x% ------------------
.\CUDAMatrixSolver.exe test.txt
@echo ------------------ %x% ------------------
@echo ""

for /l %%y in (100, 100, 2000) do (
    .\randMatrix.exe %%y test.txt
    .\parallel.exe test.txt
    @echo ------------------ %%y ------------------
    .\CUDAMatrixSolver.exe test.txt
    @echo ------------------ %%y ------------------
    @echo ""
)
