
// GPU kernel
__global__ void summation_kernel(int data_size, float * data_out)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < data_size) data_out[id] += ((id % 2 != 0) ? -1:1) / (id + 1.0);
	//	printf("%d>%f\n",id, data_out[id]);
}

__global__ void reduce(int data_size, float* data_in, float* data_out) {
	extern __shared__ float sdata[];
	
	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = (id < data_size) ? data_in[id] : 0;
	__syncthreads();

	for(unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;

		if(index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	if(tid == 0) data_out[blockIdx.x] = sdata[0];
}
