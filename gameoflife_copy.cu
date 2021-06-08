#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <cuda_runtime.h>

#define MAX_THREAD  24
/*
     0123456789
0    oooooooooo   
1    o@@@@@@@@o      Problem solving example
2    o@@@@@@@@o      8 * 6 matrix
3    o@@@@@@@@o      + 1px padding around matrix
4    o@@@@@@@@o      and use 3 * 3 filter
5    o@@@@@@@@o
6    o@@@@@@@@o
7    oooooooooo
*/

int nprocs, display, gen, width, height;
int** arr;
int** tmp;
pthread_barrier_t tbarrier;
struct timespec begin, end;

//CUDA
void dump(); 

//single & multi thread
int setPixel(int x, int y);
void* Thread(void *args);
void nextGenPixel(int start, int end, int width);
void copyAndResetData(int start, int end, int width);

typedef struct{
    int start;
    int end;
} bound;

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

__global__ void my_kernel(int *mem, int *tmp, int height, int width, int gen){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
    printf("width %d\n", blockDim.x);
	
    for(int i=0; i<gen; i++){
        if(blockIdx.x == 0 || blockIdx.x == height-1 || threadIdx.x == 0 || threadIdx.x == width-1){
            //Do nothing
        }else{
            tmp[index] = cudaNeighbor(mem, index, width);
        }
        mem[index] = tmp[index];
    }
}
int main(int argc, char *argv[]){
    pthread_t thread[MAX_THREAD];
    FILE *fp;
    char buffer[20];
    int x, y, size, length;
    //This is for convert 2d array to 1d
    int *mat_1d, *mat_1d_tmp;
    int *cuda_mem, *cuda_tmp;

    char *x_map, *y_map;

    if(argc!=7){
        printf("Parameter Error!\n");
        printf("./glife <input file> <display> <nprocs> <# of generation> <width> <height>\n");
        exit(1);
    }

    display = atoi(argv[2]);
    nprocs = atoi(argv[3]);
    gen = atoi(argv[4]);
    width = atoi(argv[5]);
    height = atoi(argv[6]);

    //Make matrix
    arr = (int**)malloc(sizeof(int*) * (height+2));
    for(int i=0; i<height+2; i++){
        arr[i] = (int*)malloc(sizeof(int) * (width+2));
    }
    tmp = (int**)malloc(sizeof(int*) * (height+2));
    for(int i=0; i<height+2; i++){
        tmp[i] = (int*)malloc(sizeof(int) * (width+2));
    }
    
    //length = (height+2) * (width+2);
    size = (height+2) * (width+2) * sizeof(int);

    mat_1d = (int*)malloc(size);
    mat_1d_tmp = (int*)malloc(size);

    //Initialize
    for(int a=0; a<height+2; a++){
        for(int b=0; b<width+2; b++){
            arr[a][b] = 0;
            tmp[a][b] = 0;
            mat_1d[a*(width+2)+b] = 0;
            mat_1d_tmp[a*(width+2)+b] = 0;
        }
    }

    if((fp=fopen(argv[1],"r")) == NULL){
        fprintf(stderr, "error");
        exit(2);
    }
    //Mapping
    while(fgets(buffer, 20, fp) != NULL){
        y_map = strtok(buffer, " ");
        x_map = strtok(NULL, " ");

        y = atoi(y_map);        
        x = atoi(x_map);

        arr[x][y] = 1;
        mat_1d[x*(width+2) +y] = 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &begin);

    if(nprocs == 0){
        //CUDA

        cudaMalloc(&cuda_mem, size);
        cudaMalloc(&cuda_tmp, size);

        cudaMemcpy(cuda_mem, mat_1d, size, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_tmp, mat_1d_tmp, size, cudaMemcpyHostToDevice);

        //Kernel code
        my_kernel<<<  height+2 , width+2  >>>(cuda_mem, cuda_tmp, height+2, width+2, gen);

        cudaMemcpy(mat_1d, cuda_mem, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(mat_1d_tmp, cuda_tmp, size, cudaMemcpyDeviceToHost);

        for(int i=0;i<height+2;i++){
            for(int j=0;j<width+2;j++){
                arr[i][j] = mat_1d[i*(width+2) +j ];
            }
        }
    }else{
	
        //SINGLE AND MULTI THREAD
        //Divide height into nprocs pieces
        bound section[MAX_THREAD];
        int x = 0;
        int y = 0;
        int div = height/nprocs;

        for(int k=0; k<nprocs; k++){
            if(k == (nprocs-1)){
                y = height;
                section[k].start = x;
                section[k].end = y;
            }else{
                y+=div;
                section[k].start = x;
                section[k].end = y;
                x+=div;
            }
        }

		pthread_barrier_init(&tbarrier, NULL, nprocs);

        for(int i=0; i<nprocs; i++){
            pthread_create(&thread[i], NULL, Thread, &section[i]);
        }

        for(int j=0; j<nprocs; j++){
            pthread_join(thread[j], NULL);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
	printf("Execution time : %2.3f sec\n",(end.tv_sec - begin.tv_sec)+(end.tv_nsec-begin.tv_nsec)/1000000000.0);

    if(display == 1){
        dump();
    }
    
    pthread_barrier_destroy(&tbarrier);

    free(arr);
    free(tmp);
    free(mat_1d);
    free(mat_1d_tmp);
    cudaFree(cuda_mem);
    cudaFree(cuda_tmp);

    return 0;
}

void *Thread(void *args){
    //get args with struct
    bound *section = (bound*)args;
    
    for(int i=0; i<gen; i++){

        nextGenPixel(section[0].start, section[0].end, width);
		pthread_barrier_wait(&tbarrier);

        copyAndResetData(section[0].start, section[0].end, width);
        pthread_barrier_wait(&tbarrier);
    }
}

void nextGenPixel(int start, int end, int wdth){
    int head = start;
    int tail = end;
    if(head == 0){
        head = 1;
    }
    if(tail == height){
        tail++;
    }
    for(int i=head; i<tail; i++){
        for(int j=1; j<=wdth; j++){
            tmp[i][j]=setPixel(i,j);
        }
    }
}

void copyAndResetData(int start, int end, int wdth){
    int tail = end;
    if(tail == height){
        tail +=2;
    }
    for(int a=start; a<tail; a++){
        for(int b=0; b<wdth+2; b++){
            arr[a][b] = tmp[a][b];
            tmp[a][b] = 0;
        }
    }
}

int setPixel(int x, int y){
    //live cell with 2 or 3 neighbors -> keep live
    //dead cell with 3 neighbors -> revive
    //other cases : die
    int current = arr[x][y];
    int neighbor = arr[x-1][y-1]+arr[x][y-1]+arr[x+1][y-1]+arr[x+1][y]+arr[x+1][y+1]+arr[x][y+1]+arr[x-1][y+1]+arr[x-1][y];
    
    if((current == 1 && neighbor == 2) || (current == 1 && neighbor == 3) || (current == 0 && neighbor == 3)){
        return 1;
    }else{
        return 0;
    }
}

void dump(){
    //   print arr info
    printf("%d x %d matrix\n", width, height);
    printf("========================================\n");
    for(int a=1; a<=height; a++){
        for(int b=1; b<=width; b++){
            if(arr[a][b]==1){
                printf("o");
            }else{
                printf("-");
            }
        }
        printf("\n");
    }
    printf("========================================\n");    
}
