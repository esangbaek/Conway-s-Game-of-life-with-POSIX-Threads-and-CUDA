#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>


pthread_barrier_t barrier;

void* wait(void *args){
	int n = *(int *)args;
    printf("Before barrier\n");
	printf("-->%d\n",n);
    pthread_barrier_wait(&barrier);
    printf("After barrier\n");
}

int main(int argc, char *argv[]){
    int num = atoi(argv[1]);
	printf("num : %d\n",num);
    pthread_t thread[4];
    pthread_barrier_init(&barrier, NULL, num);
	
	int *a = malloc(sizeof(int*)*10);
    for(int i=0;i<num;i++){
		a[i] = i;
		printf("create %d\n", a[i]);
        pthread_create(&thread[i], NULL, wait, &a[i]);
    }

	for(int j=0; j<num; j++){
		if(pthread_join(thread[j],NULL) != 0){
			printf("join error\n");
		}
	}

    pthread_barrier_destroy(&barrier);
   free(a); 
    return 0;
}
