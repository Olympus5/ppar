
/*
* Conway's Game of Life
*
* A. Mucherino
*
* PPAR, TP4
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

int N = 32;
int itMax = 1000;

// allocation only
unsigned int* allocate()
{
   return (unsigned int*)calloc(N*N,sizeof(unsigned int));
}

// conversion cell location : 2d --> 1d
// (row by row)
int code(int x,int y,int dx,int dy)
{
   int i = (x + dx)%N;
   int j = (y + dy)%N;
   if (i < 0)  i = N + i;
   if (j < 0)  j = N + j;
   return i*N + j;
}

// writing into a cell location
void write_cell(int x,int y,unsigned int value,unsigned int *world)
{
   int k;
   k = code(x,y,0,0);
   world[k] = value;
}

// reading a cell
int read_cell(int x,int y,int dx,int dy,unsigned int *world)
{
   int k = code(x,y,dx,dy);
   return world[k];
}

// updating counters
void update(int x,int y,int dx,int dy,unsigned int *world,int *nn,int *n1,int *n2)
{
   unsigned int cell = read_cell(x,y,dx,dy,world);
   if (cell != 0)
   {
      (*nn)++;
      if (cell == 1)
      {
         (*n1)++;
      }
      else
      {
         (*n2)++;
      }
   }
}

// looking around the cell
void neighbors(int x,int y,unsigned int *world,int *nn,int *n1,int *n2)
{
   int dx,dy;

   (*nn) = 0;  (*n1) = 0;  (*n2) = 0;

   // same line
   dx = -1;  dy = 0;   update(x,y,dx,dy,world,nn,n1,n2);
   dx = +1;  dy = 0;   update(x,y,dx,dy,world,nn,n1,n2);

   // one line down
   dx = -1;  dy = +1;  update(x,y,dx,dy,world,nn,n1,n2);
   dx =  0;  dy = +1;  update(x,y,dx,dy,world,nn,n1,n2);
   dx = +1;  dy = +1;  update(x,y,dx,dy,world,nn,n1,n2);

   // one line up
   dx = -1;  dy = -1;  update(x,y,dx,dy,world,nn,n1,n2);
   dx =  0;  dy = -1;  update(x,y,dx,dy,world,nn,n1,n2);
   dx = +1;  dy = -1;  update(x,y,dx,dy,world,nn,n1,n2);
}

// computing a new generation
short newgeneration(unsigned int *world1,unsigned int *world2,int xstart,int xend)
{
   int x,y;
   int nn,n1,n2;
   unsigned int cell;
   short change = 0;

   // cleaning destination world
   for (x = 0; x < N; x++)
   {
      for (y = 0; y < N; y++)
      {
         write_cell(x,y,0,world2);
      }
   }

   // generating the new world
   for (x = xstart; x < xend; x++)
   {
      for (y = 0; y < N; y++)
      {
         cell = read_cell(x,y,0,0,world1);
         neighbors(x, y, world1, &nn, &n1, &n2);

         if(nn > 3 || nn < 2) {
            // ILS MEURENT !!!!
            change = 1;
            write_cell(x,y,0,world2);
         } else {
            // ILS VIVENT !!!!
            if(cell == 0 && nn == 3) {
               // ILS NAISSENT !!!
               change = 1;
               if(n1 > 2) {
                  // DES RONDS
                  write_cell(x,y,1,world2);
               } else {
                  // DES CROIX
                  write_cell(x,y,2,world2);
               }
            } else {
               // ILS PESENT DANS LE GAME OF LIFE !!!!!!
               write_cell(x,y,cell,world2);
            }
         }
      }
   }
   return change;
}

// cleaning the screen
void cls()
{
   int i;
   for (i = 0; i < 10; i++)
   {
      fprintf(stdout,"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
   }
}

// diplaying the world
void print(unsigned int *world)
{
   int i;
   cls();
   for (i = 0; i < N; i++)  fprintf(stdout,"-");

   for (i = 0; i < N*N; i++)
   {
      if (i%N == 0)  fprintf(stdout,"\n");
      if (world[i] == 0)  fprintf(stdout," ");
      if (world[i] == 1)  fprintf(stdout,"o");
      if (world[i] == 2)  fprintf(stdout,"x");
   }
   fprintf(stdout,"\n");

   for (i = 0; i < N; i++)  fprintf(stdout,"-");
   fprintf(stdout,"\n");
   sleep(1);
}


// random generation
unsigned int* initialize_random()
{
   int x,y;
   unsigned int cell;
   unsigned int *world;

   srand(time(NULL));

   world = allocate();
   for (x = 0; x < N; x++)
   {
      for (y = 0; y < N; y++)
      {
         if (rand()%5 != 0)
         {
            cell = 0;
         }
         else if (rand()%2 == 0)
         {
            cell = 1;
         }
         else
         {
            cell = 2;
         };
         write_cell(x,y,cell,world);
      }
   }
   return world;
}

// dummy generation
unsigned int* initialize_dummy()
{
   int x,y;
   unsigned int *world;

   world = allocate();
   for (x = 0; x < N; x++)
   {
      for (y = 0; y < N; y++)
      {
         write_cell(x,y,x%3,world);
      }
   }
   return world;
}

// "glider" generation
unsigned int* initialize_glider()
{
   int x,y,mx,my;
   unsigned int *world;

   world = allocate();
   for (x = 0; x < N; x++)
   {
      for (y = 0; y < N; y++)
      {
         write_cell(x,y,0,world);
      }
   }

   mx = N/2 - 1;  my = N/2 - 1;
   x = mx;      y = my + 1;  write_cell(x,y,1,world);
   x = mx + 1;  y = my + 2;  write_cell(x,y,1,world);
   x = mx + 2;  y = my;      write_cell(x,y,1,world);
   y = my + 1;  write_cell(x,y,1,world);
   y = my + 2;  write_cell(x,y,1,world);

   return world;
}

// "small exploder" generation
unsigned int* initialize_small_exploder()
{
   int x,y,mx,my;
   unsigned int *world;

   world = allocate();
   for (x = 0; x < N; x++)
   {
      for (y = 0; y < N; y++)
      {
         write_cell(x,y,0,world);
      }
   }

   mx = N/2 - 2;  my = N/2 - 2;
   x = mx;      y = my + 1;  write_cell(x,y,2,world);
   x = mx + 1;  y = my;      write_cell(x,y,2,world);
   y = my + 1;  write_cell(x,y,2,world);
   y = my + 2;  write_cell(x,y,2,world);
   x = mx + 2;  y = my;      write_cell(x,y,2,world);
   y = my + 2;  write_cell(x,y,2,world);
   x = mx + 3;  y = my + 1;  write_cell(x,y,2,world);

   return world;
}

// main
int main(int argc,char *argv[])
{
   int it,change;
   unsigned int *world1,*world2;
   unsigned int *worldaux;

   // some useful variable for MPI program
   int my_rank, n, first_index;
   MPI_Request request;
   MPI_Status status;

   // Init MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &n);

   // 1. all processes verify that N is divisible by p: if not, the execution is aborted
   if(N < n) {
      fprintf(stderr, "It cann't have more processes than the size of your game.");
      fprintf(stderr, "Give a correct number of processes (lower or equals than %d).", N);
      fprintf(stderr, "The game abort.");
      exit(-1);
   }

   if(my_rank == 0) {
      // getting started

      // 2. process 0 generates the initial world
      world1 = initialize_random();

      // 3. process 0 sends to all other processes the generated initial world (communication type: one-to-all)
      MPI_Bcast(world1, n, MPI_INT, 0, MPI_COMM_WORLD);

      // 4. process 0 prints the initial world on the screen
      print(world1);
   }

   world2 = allocate();

   // 5. every process computes the first and last row index of its world region;
   first_index = (N*N)/n * my_rank; // THe last index is: first_index + (N/n) - 1

   it = 0;  change = 1;

   // 6. in the main while loop
   while (change && it < itMax)
   {
      // 6.a every process invokes newgeneration with its first and last row index
      change = newgeneration(world1, world2, first_index, first_index + (N * N / n) - 1);
      // 6.b the pointers of world1 and world2 are inverted, as in the sequential version
      worldaux = world1;  world1 = world2;  world2 = worldaux;
      // 6.c the processes exchange the neighbouring rows, necessary for computing the next generation (communication type: one-to-one)
      printf("(START) my rank is: %d\n", my_rank);
      if(my_rank < n-1) {
         MPI_Isend(world1, N*N, MPI_INT, my_rank + 1, 0, MPI_COMM_WORLD, &request);
         MPI_Irecv(world2, N*N, MPI_INT, my_rank + 1, 0, MPI_COMM_WORLD, &request);
      }

      if(my_rank > 0) {
         MPI_Isend(world1, N*N, MPI_INT, my_rank - 1, 0, MPI_COMM_WORLD, &request);
         MPI_Irecv(world2, N*N, MPI_INT, my_rank - 1, 0, MPI_COMM_WORLD, &request);
      }

      //We will update the left neighbor part
      if(my_rank > 0) {
         for(int i = first_index - ((N * N) / n); i < first_index; i++) {
            world1[i] = world2[i];
         }
      }

      //We will update the right neighbor part
      if(my_rank < n - 1) {
         for(int i = first_index - ((N * N) / n); i < first_index; i++) {
            world1[i] = world2[i];
         }
      }

      printf("(END) my rank is: %d\n", my_rank);

      it++;
   }

   /*
    * process 0 collects the results obtained by the other processes (communication type: all-to-one):
    * consider that the partial results are stored in different regions of different torus representations,
    * and that memory for representing the entire torus was allocated by all processes
    */
   MPI_Gather(world1, N*N, MPI_INT, world2, N*N, MPI_INT, 0, MPI_COMM_WORLD);

   // process 0 prints the final result
   print(world1);

   // ending
   free(world1);   free(world2);
   return 0;
}
