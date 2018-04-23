#include "stdlib.h"
#include "stdio.h"
#include "stdbool.h"
#include "omp.h"
#include "string.h"

#define SIZE 26
#define BUFFER_SIZE 1024

int main(int argc, char* argv[]) {
    FILE* file;
    int letters[SIZE];
    char buff[BUFFER_SIZE];
    int i, consonants = 0, vowels = 0;
    int isConsonant;

    for(i = 0; i < SIZE; i++) {
        letters[i] = 0;
    }

    #pragma omp parallel private(i) shared(letters, buff)
    {

        if(omp_get_thread_num() == 0) {
            file = fopen("/home/erwan/Cours/m1/ppar/revision/tp2/text", "r");
            fgets(buff, BUFFER_SIZE, file);

            printf("%s\n", buff);
        }

        #pragma omp barrier

        printf("Bite\n %s\n", buff);
        #pragma omp for schedule(static)
        for(i = 0; i < strlen(buff); i++) {
            if(buff[i] - 97){
                letters[buff[i] - 97]++;
            }
        }
    }

    #pragma omp parallel private(i, isConsonant) shared(consonants, vowels)
    {
        #pragma omp for schedule(static)
        for(i = 0; i < strlen(buff); i++) {
            isConsonant = (buff[i] != 'a') && (buff[i] != 'e') && (buff[i] != 'i') && (buff[i] != 'o') && (buff[i] != 'u') && (buff[i] != 'y');

            if(isConsonant) {
                #pragma omp atomic
                consonants++;
            } else {
                #pragma omp atomic
                vowels++;
            }
        }
    }

    printf("It have: \n");

    for(i = 0; i < SIZE; i++) {
        printf("%d %c\n", letters[i], (i+97));
    }

    printf("%d consonants\n", consonants);
    printf("%d vowels\n", vowels);
    printf("%d letters\n", (consonants + vowels));



    return 0;
}
