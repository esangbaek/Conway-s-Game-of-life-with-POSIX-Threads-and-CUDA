__device__ int cudaNeighbor(int *mem, int index, int width){
    //live cell with 2 or 3 neighbors -> keep live
    //dead cell with 3 neighbors -> revive
    //other cases : die
    int current = mem[index];
    int neighbor = mem[index-width-1]+mem[index-width]+mem[index-width+1]+
                    mem[index-1]+mem[index+1]+mem[index+width-1]+mem[index+width]+mem[index+width+1];
    
    if((current == 1 && neighbor == 2) || (current == 1 && neighbor == 3) || (current == 0 && neighbor == 3)){
        return 1;
    }else{
        return 0;
    }
}

__global__ void my_kernel(int *cuda_mem, int *cuda_tmp, int height, int width, int gen){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(blockIdx.x == 0 || blockIdx.x == height-1 || threadIdx.x == 0 || threadIdx.x == width-1){
            //Do nothing
        }else{
           cuda_tmp[index] = cudaNeighbor(cuda_mem, index, width);
        }
	__syncthreads();
        cuda_mem[index] = cuda_tmp[index];
	__syncthreads();
	cuda_tmp[index] = 0;
	__syncthreads();
}