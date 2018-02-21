CC=gcc
CFLAGS=-g -W -Wall -pedantic -I inc
LDFLAGS=-lm
EXEC=bin/gamelife

all: $(EXEC)

bin/gamelife: obj/gamelife.o
	$(CC) -o bin/gamelife obj/gamelife.o $(LDFLAGS)

obj/gamelife.o: src/gamelife.c
	$(CC) -o obj/gamelife.o -c src/gamelife.c $(CFLAGS)

clean:
	rm -rf *.o

mrproper: clean
	rm -rf $(EXEC)
