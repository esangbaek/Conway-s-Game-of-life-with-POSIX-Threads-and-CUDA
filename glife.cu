#ifdef __cplusplus
extern "C++" {
#include "glife.h"
}
#include <cuda.h>

// HINT: YOU CAN USE THIS METHOD FOR ERROR CHECKING
// Print error message on CUDA API or kernel launch
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
        } \
    } while (0)

// TODO: YOU MAY NEED TO USE IT OR CREATE MORE
__device__ int getNeighbors(int* grid, int tot_rows, int tot_cols,
        int row, int col) {
    return numOfNeighbors;
}

// TODO: YOU NEED TO IMPLEMENT KERNEL TO RUN ON GPU DEVICE 
__global__ void kernel()
{

}

// TODO: YOU NEED TO IMPLEMENT TO PRINT THE INDEX RESULTS 
void cuda_dump()
{
    printf("===============================\n");

    printf("===============================\n");
}

// TODO: YOU NEED TO IMPLEMENT TO PRINT THE INDEX RESULTS 
void cuda_dump_index()
{
    printf(":: Dump Row Column indices\n");
}

// TODO: YOU NEED TO IMPLEMENT ON CUDA VERSION
uint64_t runCUDA(int rows, int cols, int gen, 
                 GameOfLifeGrid* g_GameOfLifeGrid, int display)
{
    cudaSetDevice(0); // DO NOT CHANGE THIS LINE 

    uint64_t difft;

    // ---------- TODO: CALL CUDA API HERE ----------


    // Start timer for CUDA kernel execution
    difft = dtime_usec(0);
    // ----------  TODO: CALL KERNEL HERE  ----------


    // Finish timer for CUDA kernel execution
    difft = dtime_usec(difft);

    // Print the results
    if (display) {
        cuda_dump();
        cuda_dump_index();
    }
    return difft;
}
#endif
