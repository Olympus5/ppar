
// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy,
    unsigned int domain_x, unsigned int domain_y)
{
    x = (unsigned int)(x + dx) % domain_x;	// Wrap around
    y = (unsigned int)(y + dy) % domain_y;
    return source_domain[y * domain_x + x];
}

// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain,
    int domain_x, int domain_y)
{
    extern __shared__ sdata[];

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y;

	sdata[ty * domain_x + tx] = source_domain[ty * domain_x + tx];
	__syncthreads();

    // Read cell
    int myself = read_cell(sdata, tx, ty, 0, 0,
	                       domain_x, domain_y);
    	int blue = 0; // Nombre de pion bleu
	int red = 0; // Nombre de pion rouge
	int total = 0; // Nombre total de voisin
	int current; // Position initial du voisin i
	int dx; // Decalage en x
	int dy; // Decalage en y
    // TODO: Read the 8 neighbors and count number of blue and red

	//TODO: NE SUPPRIME RIEN - ERWAN
	for(int x = 0; x < 3; x++) {
		dx = (tx + x) % domain_x;
		for(int y = 0; y < 3; y++) {
			dy = (ty + y) % domain_y;
			if(dy != 0 || dx != 0) {
				current = read_cell(sdata, tx, ty, dx, dy, domain_x, domain_y);

				if(current == 2) {
					blue++;
				} else if(current == 1){
					red++;
				}
			}
		}
	}
    __syncthreads();

	total = blue + red;
	// TODO: Compute new value
	// TODO: NE SUPPRIME TOUJOURS RIEN - ERWAN
	if(total < 2 ||total > 3 ||(myself == 0&&total != 3)) {//Je meurs !!!!!!!
		myself = 0;
	} else {// Je vie
		if(myself == 0) {
			myself = (red > blue) ? 1 : 2;
		}
	}
	// TODO: Write it in dest_domain
	// TODO: NE SUPPRIME TOUJOURS RIEN - ERWAN
	dest_domain[ty * domain_x + tx] = myself;
}
