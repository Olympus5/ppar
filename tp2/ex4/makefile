CC=gcc
CFLAGS=-fopenmp -g -W -Wall -pedantic -I inc
LDFLAGS=-fopenmp -lm
EXEC=bin/main

all: $(EXEC)

bin/main: obj/main.o
	$(CC) -o bin/main obj/main.o $(LDFLAGS)

obj/main.o: src/main.c inc/main.h
	$(CC) -o obj/main.o -c src/main.c $(CFLAGS)

clean:
	rm -rf *.o

mrproper: clean
	rm -rf $(EXEC)
