
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
int itMax = 20 ;

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
         }
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

// main
int main(int argc,char *argv[])
{
   int it,change;
   unsigned int *world1,*world2;
   unsigned int *worldaux;
   int rank, nprocs, first_index, last_index, top_rank, bottom_rank;
   MPI_Status status[4];
   MPI_Request request[4];

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   if(N % nprocs != 0) {
      MPI_Abort(MPI_COMM_WORLD, -1);
      exit(-1);
   }

   if(rank == 0) {
      // getting started
      //world1 = initialize_dummy();
      //world1 = initialize_random();
      world1 = initialize_glider();
      //world1 = initialize_small_exploder();
      print(world1);
   } else {
      world1 = allocate();
   }

   world2 = allocate();

   MPI_Bcast(world1, N*N, MPI_INT, 0, MPI_COMM_WORLD);

   first_index = rank * N/nprocs;
   last_index = (rank * N/nprocs + (N/nprocs - 1));
   top_rank = (rank == 0) ? nprocs - 1 : rank - 1;
   bottom_rank = (rank == nprocs - 1) ? 0 : rank + 1;

   it = 0;  change = 1;
   while (change && it < itMax)
   {
      change = newgeneration(world1,world2,first_index,last_index);
      worldaux = world1;  world1 = world2;  world2 = worldaux;

      MPI_Isend(&world1[first_index % N], N, MPI_INT, top_rank, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Isend(&world1[last_index % N], N, MPI_INT, bottom_rank, 0, MPI_COMM_WORLD, &request[1]);
      MPI_Irecv(&world1[(first_index - 1) % N], N, MPI_INT, top_rank, 0, MPI_COMM_WORLD, &request[2]);
      MPI_Irecv(&world1[(last_index + 1) % N], N, MPI_INT, bottom_rank, 0, MPI_COMM_WORLD, &request[3]);
      MPI_Waitall(4, request, status);
      it++;
   }

   MPI_Gather(&world1[first_index % N], N/nprocs * N, MPI_INT, world2, N/nprocs * N, MPI_INT, 0, MPI_COMM_WORLD);

   if(rank == 0) {
      print(world1);
      free(world1);   free(world2);
   }

   MPI_Finalize();
   // ending
   return 0;
}
