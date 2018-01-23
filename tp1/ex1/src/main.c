#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char* argv[]) {
    unsigned int k;
    unsigned int i;
    unsigned int j;
    unsigned int n;
    unsigned int** tab;

    /* On demande à l'utilisateur d'entrer un entier naturel */
    if(argc < 2) {
        printf("Insert a natural integer please: ");
        scanf("%u", &k);
    } else {
        k = atoi(argv[1]);
    }

    /* On regarde si on a bien un entier naturel */
    if(k < 1) {
        fprintf(stderr, "Insert a correct value.\n");
        return -1;
    }

    /* On calcul le carré de la valeur entrée par l'utilisateur */
    n = 1;

    for(i = 0; i < k; i++) {
        n *= 2;
    }

    /* On alloue la mémoire pour stocker dan un tableau les n valeur compris entre 1 et le carré de la valeur entrée par l'utilisateur */
    tab = malloc(n * (sizeof (int*)));

    /* On initialise notre tableau */
    for(i = 0; i < n; i++) {
        tab[i] = malloc(2 * sizeof (int));

        /* Initialisation de la valeur */
        tab[i][0] = i + 1;

        /* Initialisation du suivant */
        if(i < n-1) {
            tab[i][1] = i + 1;
        } else {
            tab[i][1] = 0;
        }
    }

    /* On somme les valeurs du tableau entre elles */
    for(i = 0; i < (unsigned int) log2(n); i++) {
        for(j = 0; j < n; j++) {
            /* Si il y a un suivant alors on travail dessus */
            if(tab[j][1]) {
                tab[j][0] = tab[j][0] + tab[tab[j][1]][0];
                tab[j][1] = tab[tab[j][1]][1];
            }
        }
    }

    /* On affiche le résultat */
    printf("Result = %u\n", tab[0][0]);

    /* Ne pas oublier de libérer la mémoire alloué */
    for(i = 0; i < n; i++) {
        free(tab[i]);
    }

    free(tab);

    /* Tout c'est bien passé */
    return 0;
}
