#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int n, i, j, k, me, nb;
    int** m1;
    int** m2;
    int** result;

    me = 0;
    nb = 4;

    if(argc < 2) {
        printf("The size of your squared matrix: ");
        scanf("%d", &n);
    } else {
        n = atoi(argv[1]);
    }

    if(n < 1) {
        fprintf(stderr, "Error");
        return -1;
    }

    /* Initialisation des matrice */
    m1 = malloc(n * (sizeof (int*)));
    m2 = malloc(n * (sizeof (int*)));
    result = malloc(n * (sizeof (int*)));

    for(i = 0; i < n; i++) {
        m1[i] = malloc(n * (sizeof (int)));
        m2[i] = malloc(n * (sizeof (int)));
        result[i] = malloc(n * (sizeof (int)));
    }

    srand(time(NULL));

    for(i = 0; i < n; i++)  {
        for(j = 0; j < n; j++) {
            m1[i][j] = rand()%(21 - 0) + 0;
            m2[i][j] = rand()%(21 - 0) + 0;
            result[i][j] = 0;
        }
    }

    omp_set_num_threads(4);

    #pragma omp parallel private(i, j, k) shared(n, m1, m2, result)
    {
        #pragma omp for schedule(dynamic)
        for(i = 0; i < n; i++) {
            for(j = 0; j < n; j++) {
                for(k = 0; k < n; k++) {
                    result[i][j] += m1[i][k] * m2[k][j];
                }
            }
        }
    }

    /* Affiche le résultat de la multiplication */
    printf("Matrice A:\n\n");

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%d\t", m1[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrice B:\n\n");

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%d\t", m2[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrice C: \n\n");

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%d\t", result[i][j]);
        }
        printf("\n");
    }

    /* Ne pas oublier de libérer la mémoire */
    for(i = 0; i < n; i++) {
        free(m1[i]);
        free(m2[i]);
        free(result[i]);
    }

    free(m1);
    free(m2);
    free(result);

}
