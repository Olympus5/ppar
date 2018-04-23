#include "stdlib.h"
#include "stdio.h"
#include "stdbool.h"
#include "omp.h"

#define SIZE 1000

int main(int argc, char* argv[]) {
    bool b[SIZE];
    int i, j, nb = 0;

    for(i = 0; i < SIZE; i++) {
        b[i] = true;
    }

    #pragma omp parallel private(i,j) shared(b)
    {
        #pragma omp for schedule(static)
        for(i = 2; i < SIZE; i++) {
            if(b[i] == true) {
                for(j = i*2; j < SIZE; j+=i) {
                    b[j] = false;
                }
            }
        }
    }

    for(i = 2; i < SIZE; i++) {
        if(b[i] == true) {
            nb++;
        }
    }

    printf("He have %d primary number\n", nb);

    return 0;
}
