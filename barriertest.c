#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>


pthread_barrier_t barrier;

void* wait(void *args){
    printf("Before barrier\n");
    pthread_barrier_wait(&barrier);
    printf("After barrier\n");
}

int main(int argc, char *argv[]){
    int num = atoi(argv[1]);

    pthread_t thread[8];
    pthread_barrier_init(&barrier, NULL, num);

    for(int i=0;i<num;i++){
        pthread_create(&thread[i], NULL, wait, NULL);
    }

    pthread_barrier_destroy(&barrier);
    
    return 0;
}