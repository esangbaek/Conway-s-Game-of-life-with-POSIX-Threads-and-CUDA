# Conway's Game of life with POSIX Threads and CUDA

### Multi thread    
스레드 개수에 따른 실행시간 비교    
./glife sample_inputs/make-a_71_81 0 N 1000 400 400		( N = nprocs )    
(20회 실행하여 평균)
| nprocs | Seconds |      
|-----|-----|     
| 1 | 6.16 |    
| 2 | 2.87 |
| 4 | 1.34 |
| 8 | 0.71 |
| 16 | 0.68 |


### CUDA    
CUDA implementation    
./glife sample_inputs/make-a_71_81 0 0 1000 400 400    
(30회 실행하여 평균)    
|CUDA 실행시간|cudaMalloc|cudaMemcpy(Host to Device)|kernel|cudaMemcpy(Device to Host)|
|-----|-----|-----|-----|-----|
|0.967|0.946|≈ 0|0.020|≈ 0|    



#### Compile     
    $ /usr/local/cuda/bin/nvcc -o glife gameoflife.cu –w
