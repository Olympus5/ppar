#include "stdlib.h"
#include "stdio.h"
#include "omp.h"

int main(int argc, char* argv[]) {
    int me = 0;
    int nb = 4;

    omp_set_num_threads(nb);

    #pragma omp parallel private(me, nb)
    {
        nb = omp_get_num_threads();
        me = omp_get_thread_num();

        printf("Hello from thread %d\n", me);

        if(me == 0) {
            printf("We are a groupd of %d threads.\n", nb);
        }
    }

    return 0;
}
