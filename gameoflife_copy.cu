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

//void CUDA();
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

struct timespec begin, end;


__device__ int setCudaPixel(int x, int y, int **mem){
    //live cell with 2 or 3 neighbors -> keep live
    //dead cell with 3 neighbors -> revive
    //other cases : die
    int current = mem[x][y];
    int neighbor = mem[x-1][y-1]+mem[x][y-1]+mem[x+1][y-1]+mem[x+1][y]+mem[x+1][y+1]+mem[x][y+1]+mem[x-1][y+1]+mem[x-1][y];
    
    if((current == 1 && neighbor == 2) || (current == 1 && neighbor == 3) || (current == 0 && neighbor == 3)){
        return 1;
    }else{
        return 0;
    }
}

__global__ void my_kernel(int *mem, int *tmp, int height, int width){
    for(int i=1; i<=height; i++){
        for(int j=1; j<=width; j++){
            tmp[i][j]=setCudaPixel(i,j, mem);
        }
    }

    for(int j=0; j< height; j++){
        for(int k=0; k<width; k++){
            mem[j][k] = tmp[j][k];
            tmp[j][k] = 0;
        }
    }
}






int main(int argc, char *argv[]){
    pthread_t thread[MAX_THREAD];
    FILE *fp;
    char buffer[20];
    int x, y, size;
    int *cuda_mem, *cuda_tmp;
    char *x_map, *y_map;

	clock_gettime(CLOCK_MONOTONIC, &begin);

    if(argc!=7){
        printf("Parameter Error!\n");
        printf("./glife <input file> <display> <nprocs> <# of generation> <width> <height>\n");
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
    size = (height+2) * (width+2) * sizeof(int);

    //Initiate
    for(int a=0; a<height+2; a++){
        for(int b=0; b<width+2; b++){
            arr[a][b]=0;
            tmp[a][b]=0;
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

        arr[x][y]=1;
    }

    if(nprocs == 0){
        //CUDA

        cudaMalloc(&cuda_mem, height+2);
        for(int i=0; i<height+2; i++){
            cuda_mem[i] = (int*)malloc(sizeof(int) * (width+2));
        }

        cudaMalloc(&cuda_tmp, width+2);
        for(int i=0; i<height+2; i++){
            cuda_tmp[i] = (int*)malloc(sizeof(int) * (width+2));
        }
        cudaMemcpy(cuda_mem, arr, size, cudaMemcpyHostToDevice);
        
        //Kernel code
        for(int i=0; i<gen; i++){
            //KERNEL CODE
            my_kernel<<<  1 , 1  >>>(cuda_mem, cuda_tmp, height, width);
            cudaDeviceSynchronize();
            
        }

        cudaMemcpy(arr, cuda_mem, size, cudaMemcpyDeviceToHost);
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
