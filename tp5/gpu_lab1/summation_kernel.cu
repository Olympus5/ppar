
// GPU kernel
__global__ void summation_kernel(int data_size, float * data_out)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < data_size) data_out[id] += ((id % 2 != 0) ? -1:1) / (id + 1.0);
//	printf("%d>%f\n",id, data_out[id]);
}

