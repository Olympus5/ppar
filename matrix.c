#include "stdlib.h"
#include "stdio.h"
#include "omp.h"
#include "time.h"

#define N 10

int main(int argc, char* argv[]) {
    int a[N][N];
    int b[N][N];
    int c[N][N];
    int i, j, k;

    srand(time(NULL));

    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            a[i][j] = rand() % (21 - 0) + 0;
            b[i][j] = rand() % (21 - 0) + 0;
            c[i][j] = 0;
        }
    }

    #pragma omp parallel shared(a, b, c) private(i, j, k)
    {
        #pragma omp for schedule(static)
        for(i = 0; i < N; i++) {
            for(j = 0; j < N; j++) {
                for(k = 0; k < N; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            printf("%d ", a[i][j]);
        }

        printf("\n");
    }

    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            printf("%d ", b[i][j]);
        }

        printf("\n");
    }

    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            printf("%d ", c[i][j]);
        }

        printf("\n");
    }

    return 0;
}
