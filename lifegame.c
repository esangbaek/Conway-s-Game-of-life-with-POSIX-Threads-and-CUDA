#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define MAX_THREAD  4   //my own laptop option
/*
##########
#@@@@@@@@#      Problem solving example
#@@@@@@@@#      8 * 8 matrix
#@@@@@@@@#      + 1px padding around matrix
#@@@@@@@@#      and use 3 * 3 filter
#@@@@@@@@#
#@@@@@@@@#
##########
*/

int nprocs, display, gen, width, height;
int** arr;
int** tmp;
pthread_barrier_t tbarrier;

void singleThread();
//void multiThread();
//void CUDA();
void dump(int height, int width); 
void nextGenPixel(int height, int width);
int setPixel(int x, int y);
void copyAndResetData(int height, int width);
void* multiThread(void *args);

int main(int argc, char *argv[]){
    pthread_t thread[MAX_THREAD];
    clock_t startTime, endTime;
    FILE *fp;
    char buffer[20];
    int x, y;
    char *x_map, *y_map;

    startTime = clock();

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
    for(int i=0;i<height+2;i++){
        arr[i] = (int*)malloc(sizeof(int) * (width+2));
    }
    tmp = (int**)malloc(sizeof(int*) * (height+2));
    for(int i=0;i<height+2;i++){
        tmp[i] = (int*)malloc(sizeof(int) * (width+2));
    }
    //Initiate
    for(int a=0;a<height+2;a++){
        for(int b=0;b<width+2;b++){
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
        y = atoi(y_map);
        x_map = strtok(NULL, " ");
        x = atoi(x_map);
        arr[x][y]=1;
    }

    if(nprocs == 0){
        //CUDA

    }else if(nprocs == 1){
        //SINGLE THREAD

        singleThread();
    }else{
        //NULTI THREAD

        for(int i=0; i<nprocs; i++){
            pthread_create(&thread[i], NULL, multiThread, NULL);
        }
        pthread_barrier_init(&tbarrier, NULL, nprocs);

        for(int j=0; j<nprocs; j++){
            pthread_join(thread[j], NULL);
        }
        //num of thread = nprocs
    }

    endTime = clock();
    printf("Execution time : %2.3f sec\n",(float)(endTime-startTime)/CLOCKS_PER_SEC);

    if(display == 1){
        dump(height, width);
    }
    
    pthread_barrier_destroy(&tbarrier);
    return 0;
}

void* multiThread(void *args){
    pthread_barrier_wait(&tbarrier);
}

void singleThread(){
    for(int i=0;i<gen;i++){
        nextGenPixel(height, width);
        copyAndResetData(height, width);
    }
}

void nextGenPixel(int height, int width){
    for(int i=1;i<=height;i++){
        for(int j=1;j<=width;j++){
            tmp[i][j] = setPixel(i, j);
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

void copyAndResetData(int height, int width){
    for(int a=0;a<height+2;a++){
        for(int b=0;b<width+2;b++){
            arr[a][b] = tmp[a][b];
            tmp[a][b]=0;
        }
    }
}

void dump(int height, int width){
    //   print arr info
    printf("%d x %d matrix\n", width, height);
    printf("========================================\n");
    for(int a=1;a<=height;a++){
        for(int b=1;b<=width;b++){
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