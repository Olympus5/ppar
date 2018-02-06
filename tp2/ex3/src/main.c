#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    char* set;
    int size, i, j, tot;

    if(argc < 2) {
        printf("Veuillez entrer un nombre (supérieur à 3): \n");
        scanf("%d", &size);
    } else {
        size = atoi(argv[1]);
    }


    if(size < 3) {
        fprintf(stderr, "Error");
        exit(-1);
    }

    set = (char*) malloc(sizeof (char) * size);

    for(i = 0; i < size; i++) {
        set[i] = 1;
    }

    set[0] = set[1] = 0;

    omp_set_num_threads(4);

    #pragma omp parallel shared(set) private(i, j)
    {
        #pragma omp for schedule(static)
        for(i = 2; i < size; i++) {
            if(set[i]) {
                for(j = i*2; j < size; j+=i) {
                    set[j] = 0;
                }
            }
        }
    }

    printf("La liste des nombre premier: \n");

    tot = 0;

    for(i = 0; i < size; i++) {
        if(set[i]) {
            printf("%d\n", i);
            tot++;
        }
    }

    printf("Il y a %d nombre premier\n", tot);

    free(set);

    return 0;
}
