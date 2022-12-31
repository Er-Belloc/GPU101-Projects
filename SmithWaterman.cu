#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define NUM		1000
#define SEQ_LEN	512

#define THREADS_PER_BLOCK 256

// ==================================================================================================================================================
// Macros
// ==================================================================================================================================================
#define CHECK_CUDA(call)																									\
						{																									\
							const cudaError_t err = call;																	\
							if (err != cudaError::cudaSuccess)																\
							{																								\
								printf("ERROR: %s in file %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);	\
								goto End;																					\
							}																								\
						}

#define CHECK_CUDA_KERNEL_CALL																								\
						{																									\
							const cudaError_t err = cudaGetLastError();														\
							if (err != cudaError::cudaSuccess)																\
							{																								\
								printf("ERROR: %s in file %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);	\
								goto End;																					\
							}																								\
						}

#define _CHECK_TIME(call, ...)										\
					{												\
						cudaEventRecord(start);						\
						call(__VA_ARGS__);							\
						CHECK_CUDA_KERNEL_CALL;						\
						cudaEventRecord(stop);						\
						cudaDeviceSynchronize();					\
						cudaEventElapsedTime(&millis, start, stop);	\
						printf("Execution time: %fms\n", millis);	\
					}

#define CHECK_TIME(call, ...)										\
					{												\
						cudaEventRecord(start);						\
						call(__VA_ARGS__);							\
						CHECK_CUDA_KERNEL_CALL;						\
						cudaEventRecord(stop);						\
						cudaDeviceSynchronize();					\
						cudaEventElapsedTime(&millis, start, stop);	\
					}


// ==================================================================================================================================================
// Penalties and Directions
// ==================================================================================================================================================
constexpr int c_insertion_penalty	= -2; // Left 
constexpr int c_deletion_penalty	= -2; // Up
constexpr int c_match_penalty		= +1; // Note: positive
constexpr int c_mismatch_penalty	= -1;

enum class directions : char
{
	invalid = 0,
	match,
	mismatch,
	deletion,
	insertion,

	total
}; // enum directions


// ==================================================================================================================================================
// Main Functions
// ==================================================================================================================================================
bool CPU_ComputeSW(const unsigned num_entries, const unsigned dim_seqs, const char* queries, const char* references,
	char* CPU_simple_rev_cigar, int* CPU_res, float& millis);

bool GPU_ComputeSW(const unsigned num_entries, const unsigned dim_seqs, const char* queries, const char* references,
	char* GPU_simple_rev_cigar, int* GPU_res, float& millis);

int main(int argc, char* argv[])
{
#ifdef SEPARATE_TRACEBACK
	printf("Warning: Compiled with -D SEPARATE_TRACEBACK\n");
#endif

	srand((unsigned)time(nullptr));

	constexpr char alphabet[] = { 'A', 'C', 'G', 'T', 'N' };
	constexpr auto num_chars = sizeof(alphabet);

	// Note: these are not C string: they're not zero terminated
	char* queries = nullptr;
	char* references = nullptr;
	char* CPU_simple_rev_cigar = nullptr;
	char* GPU_simple_rev_cigar = nullptr;
	int* CPU_res = nullptr;
	int* GPU_res = nullptr;

	queries		= (char*)malloc(NUM * SEQ_LEN * sizeof(char));
	references	= (char*)malloc(NUM * SEQ_LEN * sizeof(char));
	CPU_simple_rev_cigar = (char*)malloc(NUM * SEQ_LEN * 2 * sizeof(char));
	GPU_simple_rev_cigar = (char*)malloc(NUM * SEQ_LEN * 2 * sizeof(char));
	CPU_res = (int*)malloc(NUM * sizeof(int));
	GPU_res = (int*)malloc(NUM * sizeof(int));

	if (queries == nullptr ||
		references == nullptr ||
		CPU_simple_rev_cigar == nullptr ||
		GPU_simple_rev_cigar == nullptr ||
		CPU_res == nullptr ||
		GPU_res == nullptr)
	{
		if (GPU_res)				free(GPU_res);
		if (CPU_res)				free(CPU_res);
		if (GPU_simple_rev_cigar)	free(GPU_simple_rev_cigar);
		if (CPU_simple_rev_cigar)	free(CPU_simple_rev_cigar);
		if (references)				free(references);
		if (queries)				free(queries);
		return EXIT_FAILURE;
	} // end if

	// Randomly generate sequences
	for (unsigned i = 0; i < NUM * SEQ_LEN; ++i)
	{
		queries[i] = alphabet[rand() % num_chars];
		references[i] = alphabet[rand() % num_chars];
	} // end for

	// Compute algorithm both on a CPU and on a GPU
	float CPU_millis = 0.f;
	float GPU_millis = 0.f;
	const bool ok1 = CPU_ComputeSW(NUM, SEQ_LEN, queries, references, CPU_simple_rev_cigar, CPU_res, CPU_millis);
	const bool ok2 = GPU_ComputeSW(NUM, SEQ_LEN, queries, references, CPU_simple_rev_cigar, GPU_res, GPU_millis);
	printf("CPU: %f ms\nGPU: %f ms\nGPU is %.3f faster\n", CPU_millis, GPU_millis, CPU_millis / GPU_millis);

	// Compare results
	bool good = true;
	if (ok1 && ok2)
	{
		for (unsigned i = 0; i < NUM && good; ++i)
			good = (CPU_res[i] == GPU_res[i]);

		if (good)
			printf("Max scores correct\n");
		else
			printf("Max scores wrong\n");


		for (unsigned i = 0; i < NUM * SEQ_LEN * 2 && good; ++i)
			good = (CPU_simple_rev_cigar[i] == GPU_simple_rev_cigar[i]);

		if (good)
			printf("Tracebacks correct\n");
		else
			printf("Tracebacks wrong\n");
	} // end if

	free(GPU_res);
	free(CPU_res);
	free(GPU_simple_rev_cigar);
	free(CPU_simple_rev_cigar);
	free(references);
	free(queries);
	return 0;
} // main


// ==================================================================================================================================================
// CPU Implementation - copied and adapted from "https://github.com/albertozeni/GPU101-Projects/blob/main/smith-waterman/sw.c"
// ==================================================================================================================================================
inline int max4(const int zero, const int upleft, const int up, const int left) // 0, upleft, up, left
{
	const int tmp1 = zero > upleft	? zero : upleft;	// upleft is stronger than 0
	const int tmp2 = left > up		? left : up;		// up is stronger than left
	return tmp2 > tmp1 ? tmp2 : tmp1; // upleft is stronger than both up and left
} // max4

inline void backtrace(char* simple_rev_cigar, const directions* dir_mat, const unsigned dim_dir, unsigned i, unsigned j, const unsigned max_cigar_len)
{
	unsigned n;
	directions dir;
	for (n = 0; n < max_cigar_len && (dir = dir_mat[i * dim_dir + j]) != directions::invalid; n++)
	{
		switch (dir)
		{
		case directions::match: case directions::mismatch:	i--;	j--;	break;
		case directions::deletion:							i--;			break;
		case directions::insertion:									j--;	break;
		}
		simple_rev_cigar[n] = (char)dir;
	} // end for

	for (; n < max_cigar_len; n++)
		simple_rev_cigar[n] = (char)directions::invalid;
} // backtrace

bool CPU_ComputeSW(const unsigned num_entries, const unsigned dim_seqs, const char* queries, const char* references,
	char* CPU_simple_rev_cigar, int* CPU_res, float& millis)
{
	// Used for each computation
	int* scores = nullptr;
	directions* dirs = nullptr;

	const unsigned matrix_row = (dim_seqs + 1);
	const unsigned matrix_col = (dim_seqs + 1);
	const unsigned matrix_dim = matrix_row * matrix_col;

	scores = (int*)malloc(matrix_dim * sizeof(int));
	dirs = (directions*)malloc(matrix_dim * sizeof(directions));

	if (scores == nullptr || dirs == nullptr)
	{
		if (dirs)	free(dirs);
		if (scores)	free(scores);
		return false;
	} // end if

	const auto start = std::chrono::high_resolution_clock::now();

	for (unsigned n = 0; n < num_entries; n++)
	{
		const unsigned base_index = n * dim_seqs;

		int max = -1; // In sw all scores of the alignment are >= 0, so this will be for sure changed
		unsigned max_i = 0, max_j = 0;

		// Initialize the scoring matrix and direction matrix to 0 - adapted from github
		for (unsigned i = 0; i < matrix_dim; i++)
		{
			scores[i] = 0;
			dirs[i] = directions::invalid;
		}

		// Compute the alignment
		for (unsigned i = 1; i < matrix_row; i++)
		{
			for (unsigned j = 1; j < matrix_col; j++)
			{
				// Compare the sequences characters				
				const int comparison = (queries[base_index + i - 1] == references[base_index + j - 1]) ? c_match_penalty : c_mismatch_penalty;
				
				// Compute the cell knowing the comparison result
				const int tmp = max4(0,
					scores[(i - 1) * matrix_col + (j - 1)] + comparison, // upleft
					scores[(i - 1) * matrix_col + j] + c_deletion_penalty, // up
					scores[i * matrix_col + (j - 1)] + c_insertion_penalty); // left

				directions dir;
				if (tmp == (scores[(i - 1) * matrix_col + (j - 1)] + comparison))
					dir = (comparison == c_match_penalty ? directions::match : directions::mismatch);
				else if (tmp == (scores[(i - 1) * matrix_col + j] + c_deletion_penalty))
					dir = directions::deletion;
				else if (tmp == (scores[i * matrix_col + (j - 1)] + c_insertion_penalty))
					dir = directions::insertion;
				else // zero
					dir = directions::invalid;

				dirs[i * matrix_col + j] = dir;
				scores[i * matrix_col + j] = tmp;

				if (tmp > max)
				{
					max = tmp;
					max_i = i;
					max_j = j;
				}
			} // for columns
		} // for rows

		CPU_res[n] = scores[max_i * matrix_col + max_j];
		backtrace(&CPU_simple_rev_cigar[n * dim_seqs * 2], dirs, matrix_col, max_i, max_j, dim_seqs * 2);
	} // end for
	const auto end = std::chrono::high_resolution_clock::now();

	millis = (end - start).count() / 1000.f / 1000.f;

	free(dirs);
	free(scores);
	return true;
} // CPU_ComputeSW

// ==================================================================================================================================================
// GPU Implementation
// ==================================================================================================================================================
struct SW_MatrixEntry
{
	directions dir;
	int score;
}; // SW_MatrixEntry

__global__ void InitSW(SW_MatrixEntry* dev_entries,
	const char* dev_seqs_first, const unsigned dim1,
	const char* dev_seqs_second, const unsigned dim2);


#ifndef SEPARATE_TRACEBACK

__global__ void ComputeSW(SW_MatrixEntry* dev_entries,
	const unsigned row, const unsigned col, int* dev_res,
	char* dev_path, const unsigned dim_path);

#else

__global__ void ComputeSW(SW_MatrixEntry* dev_entries,
	const unsigned row, const unsigned col, int* dev_res, int* coords);

__global__ void Traceback(const SW_MatrixEntry* dev_entries,
	const unsigned row, const unsigned col, const int* coords,
	char* dev_path, const unsigned dim_path, const unsigned entries);
#endif


// Note: queries and references must have the same size
bool GPU_ComputeSW(const unsigned num_entries, const unsigned dim_seqs, const char* queries, const char* references,
	char* GPU_simple_rev_cigar, int* GPU_res, float& millis)
{
	bool ret = false;

	// Define number of threads and groups; on Colab each group can have up to 1024 threads
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	// ==============================================================================================================================================
	// CUDA Variables
	cudaEvent_t start, stop;
	char* dev_seq_first = nullptr;
	char* dev_seq_second = nullptr;
	char* dev_cigars = nullptr;
	int* dev_res = nullptr;

#ifdef SEPARATE_TRACEBACK
	int* dev_coords = nullptr;
#endif

	// These could theoretically be on GPU's shared memory, if only there was enough space
	SW_MatrixEntry* dev_matrix = nullptr;

	// ==============================================================================================================================================
	// Kernels
	const auto initSW = [&]()
	{
		const dim3 threadsPerBlock // (16, 16, 4)
		{
			(unsigned)(dim_seqs / prop.warpSize),
			(unsigned)(dim_seqs / prop.warpSize),
			(unsigned)(prop.maxThreadsPerBlock / (dim_seqs / prop.warpSize * dim_seqs / prop.warpSize))
		};

		const dim3 blocksPerGrid // (1, 1, 250)
		{
			(unsigned)1,
			(unsigned)1,
			(unsigned)(num_entries / threadsPerBlock.z)
		};

		InitSW <<< blocksPerGrid, threadsPerBlock >>> (dev_matrix,
			dev_seq_first, dim_seqs,
			dev_seq_second, dim_seqs);
	};

	const auto computeSW = [&]()
	{
		const dim3 threadsPerBlock // (32, 1, 8)
		{
			(unsigned)prop.warpSize,
			(unsigned)1,
			(unsigned)(THREADS_PER_BLOCK / prop.warpSize) // Keep it low since these threads needs to be synchronized
		};

		const dim3 blocksPerGrid // (1, 1, 125)
		{
			(unsigned)1,
			(unsigned)1,
			(unsigned)(num_entries / threadsPerBlock.z)
		};

		const unsigned dim_per_matrix = 3 * threadsPerBlock.x; // max_score, max_i, max_j for each thread
		const unsigned matrix_per_block = threadsPerBlock.z;
		const unsigned sharedMemReq = dim_per_matrix * matrix_per_block * sizeof(unsigned);
		ComputeSW <<< blocksPerGrid, threadsPerBlock, sharedMemReq >>> (
			dev_matrix, dim_seqs + 1, dim_seqs + 1,
			dev_res,
#ifndef SEPARATE_TRACEBACK
			dev_cigars, dim_seqs * 2
#else
			dev_coords
#endif
			);
	};

#ifdef SEPARATE_TRACEBACK
	const auto traceback = [&]()
	{
		const dim3 threadsPerBlock // (128, 1, 1)
		{
			(unsigned)(THREADS_PER_BLOCK / 2),
			(unsigned)1,
			(unsigned)1
		};

		const dim3 blocksPerGrid // (8, 1, 1)
		{
			(unsigned)std::ceil(float(num_entries) / (THREADS_PER_BLOCK / 2)),
			(unsigned)1,
			(unsigned)1
		};

		Traceback <<< blocksPerGrid, threadsPerBlock >>> (
			dev_matrix, dim_seqs + 1, dim_seqs + 1, dev_coords, dev_cigars, dim_seqs * 2, num_entries);
	};
#endif

	// Choose which GPU to run on, change this on a multi-GPU system.
	CHECK_CUDA(cudaSetDevice(0));

	// ==============================================================================================================================================
	// Allocate GPU buffers
	CHECK_CUDA(cudaMalloc((void**)&dev_seq_first,	num_entries * dim_seqs							* sizeof(dev_seq_first[0])));
	CHECK_CUDA(cudaMalloc((void**)&dev_seq_second,	num_entries * dim_seqs							* sizeof(dev_seq_second[0])));
	CHECK_CUDA(cudaMalloc((void**)&dev_cigars,		num_entries * dim_seqs * 2						* sizeof(dev_cigars[0])));
	CHECK_CUDA(cudaMalloc((void**)&dev_res,			num_entries										* sizeof(dev_res[0])));
	CHECK_CUDA(cudaMalloc((void**)&dev_matrix,		num_entries * (dim_seqs + 1) * (dim_seqs + 1)	* sizeof(dev_matrix[0])));

#ifdef SEPARATE_TRACEBACK
	CHECK_CUDA(cudaMalloc((void**)&dev_coords,		num_entries * 2									* sizeof(dev_coords[0])));
#endif


	// ==============================================================================================================================================
	// Create GPU Events to measure time
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	// ==============================================================================================================================================
	// Send Data to GPU
	CHECK_CUDA(cudaMemcpy(dev_seq_first, queries,		num_entries * dim_seqs * sizeof(dev_seq_first[0]), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(dev_seq_second, references,	num_entries * dim_seqs * sizeof(dev_seq_second[0]), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ==============================================================================================================================================
	// Execute Algorithm
	{
		float saveMillis = 0.f;
		CHECK_TIME(initSW);
		saveMillis = millis; // CHECK_TIME overrides millis
		CHECK_TIME(computeSW);

#ifdef SEPARATE_TRACEBACK
		saveMillis += millis; // CHECK_TIME overrides millis
		CHECK_TIME(traceback);
#endif

		millis += saveMillis;
	}

	// ==============================================================================================================================================
	// Retrieve Data from GPU
	CHECK_CUDA(cudaMemcpy(GPU_res,				dev_res,	num_entries					* sizeof(dev_res[0]), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(GPU_simple_rev_cigar, dev_cigars, num_entries * dim_seqs * 2	* sizeof(dev_cigars[0]), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	ret = true;

	// ==============================================================================================================================================
End:
	// Destroy CUDA Events
	if (ret) cudaEventDestroy(stop);
	if (ret) cudaEventDestroy(start);

	// Release Device Memory
#ifdef SEPARATE_TRACEBACK
	if (dev_coords)		cudaFree(dev_coords);
#endif

	if (dev_matrix)		cudaFree(dev_matrix);
	if (dev_res)		cudaFree(dev_res);
	if (dev_cigars)		cudaFree(dev_cigars);
	if (dev_seq_second)	cudaFree(dev_seq_second);
	if (dev_seq_first)	cudaFree(dev_seq_first);

	return ret;
} // GPU_ComputeSW


// ==================================================================================================================================================
// CUDA Kernels
// ==================================================================================================================================================
__global__ void InitSW(SW_MatrixEntry* dev_entries,
	const char* dev_seqs_first, const unsigned dim1,
	const char* dev_seqs_second, const unsigned dim2)
{
	const unsigned x = threadIdx.x; // blockIdx.x = 0 because gridDim.x = 1
	const unsigned y = threadIdx.y; // blockIdx.y = 0 because gridDim.y = 1
	const unsigned z = blockIdx.z * blockDim.z + threadIdx.z; // Matrix considered by this thread

	const unsigned matrix_base = z * (dim1 + 1) * (dim2 + 1);
	const unsigned base_seq1 = z * dim1;
	const unsigned base_seq2 = z * dim2;

	// Some threads will be waiting for others to complete, since they might have more cells to process than others
	for (unsigned i = x; i <= dim1; i += blockDim.x)
	{
		for (unsigned j = y; j <= dim2; j += blockDim.y)
		{
			const unsigned index_this = matrix_base + i * (dim2 + 1) + j;
			const int index_seq1 = base_seq1 + (i - 1);
			const int index_seq2 = base_seq2 + (j - 1);

			// Initialize
			const bool valid = (base_seq1 <= index_seq1) && (base_seq2 <= index_seq2);
			const bool mux_1 = valid && (dev_seqs_first[index_seq1] == dev_seqs_second[index_seq2]);
			const bool mux_2 = (i != 0) && (j != 0); // Skips first column and row
			const int res_1 = mux_1 * c_match_penalty + !mux_1 * c_mismatch_penalty;
			const int res_2 = valid * mux_2 * res_1; // No need for !mux_2*0
			dev_entries[index_this].score = res_2;
			dev_entries[index_this].dir = directions::invalid;
		}
	}
} // InitSW_

// Note: first row and first column don't need to be processed
__global__ void ComputeSW(SW_MatrixEntry* dev_entries,
	const unsigned row, const unsigned col, int* dev_res,
#ifndef SEPARATE_TRACEBACK
	char* dev_path, const unsigned dim_path
#else
	int* coords
#endif
	)
{
	const unsigned x = blockIdx.x * blockDim.x + threadIdx.x; // Use a warp (32 threads) per matrix; gridDim.x=1 so blockIdx.x is always 0
	const unsigned z = blockIdx.z * blockDim.z + threadIdx.z; // Matrix considered by this thread

	const unsigned matrix_base = z * row * col;

	// Shared by all threads in this block, automatically freed at block termination
	extern __shared__ unsigned shared_memory[]; // Final scores are nonnegative

	// Each thread mantains its own values till the end od the computation; do not mix matrices
	int max_score = -1; // All scores will be nonnegative, so this will be changed at some point
	unsigned max_i = 0, max_j = 0;

	// Compute scores: the matrix is divided into parallelograms
	const int start_j = 1 - (x % blockDim.x); // First thread processes [1][1], second [2][0], thirs [3][-1], last one [32][-30]
	const int stop_j = col + (blockDim.x + start_j - 2); // First thread stops before [1][544], second [2][543], thirs [3][542], last one [32][513]

	for (unsigned i = x + 1; i < row; i += blockDim.x) // Never goes outside the matrix, do not process first row
	{
		for (int j = start_j; j < stop_j; j++) // Might access memory before and after each row
		{
			// Since i is in range [1, 512], no matter the value of j, the following code never goes outside the matrix (threads go like "/->" )
			const bool valid = (0 < j && j < col); // This is used just to ensure that each cell is updated only at the right time

			const unsigned index_this	= matrix_base + i * col + j;
			const int upleft_index		= index_this - col - 1;	// One row up, one column left
			const int up_index			= index_this - col;		// One row up
			const int left_index		= index_this - 1;		// One column left

			// These are the possible results of the current cell
			const int upleft	= dev_entries[upleft_index].score + dev_entries[index_this].score; // out_scores[i * dim2 + j] contains cell penalty for upleft
			const int up		= dev_entries[up_index].score + c_deletion_penalty;	// Going up -> Deletion
			const int left		= dev_entries[left_index].score + c_insertion_penalty;	// Going left -> Insertion

			// Find max score		
			const bool mux_match	= 0 > upleft;
			const int max1			= !mux_match * upleft; // max(upleft, 0)
			directions dir1			= directions(!mux_match * (unsigned)directions::match); // Either match or invalid (if 0 is max), mismatch is the same as match

			const bool mux_ins_del	= left > up; // Note: deletions have higher priorities than insertions 
			const int max2			= mux_ins_del * left + !mux_ins_del * up; // max(left, up)
			const directions dir2	= directions(mux_ins_del * (unsigned)directions::insertion + !mux_ins_del * (unsigned)directions::deletion); // either ins or del 

			const bool mux_final		= max2 > max1; // Note: matches/mismatches (or zeros) have higher priorities than deletions/insertions
			const int out_score			= mux_final * max2 + !mux_final * max1; // max(max1, max2)
			const directions out_dir	= directions(mux_final * (unsigned)dir2 + !mux_final * (unsigned)dir1);

			// If j is out-of-range, this could theoretically override the correct results if another thread is updating the same cell,
			// but since all threads operating on the same matrix are always synchronized, this never happens
			dev_entries[index_this].score	= valid * out_score + !valid * dev_entries[index_this].score; // max(max1, max2)
			dev_entries[index_this].dir		= directions(valid * (unsigned)out_dir + !valid * (unsigned)dev_entries[index_this].dir);

			const bool is_greater = valid && dev_entries[index_this].score > max_score;
			max_score = is_greater * dev_entries[index_this].score + !is_greater * max_score;
			max_i = is_greater * i + !is_greater * max_i;
			max_j = is_greater * j + !is_greater * max_j;
		}
	}

	const unsigned threadIndex = 3 * (threadIdx.z * blockDim.x + x); // For x=0, this is the beginning of the area dedicated to the matrix of this group
	shared_memory[threadIndex] = (unsigned)max_score;
	shared_memory[threadIndex + 1] = max_i;
	shared_memory[threadIndex + 2] = max_j;

	// Computation done; to be able to calculate actual max score, synchronize all threads in this block
	__syncthreads();

	// Now all matrices are complete
	if (x == 0) // Only one thread per matrix will execute this
	{
		unsigned threadMaxIndex = threadIndex;
		for (unsigned i = 1; i < blockDim.x; i++)
		{
			const unsigned index = threadIndex + (i * 3);
			const unsigned newMax = shared_memory[index];
			const unsigned oldMax = shared_memory[threadMaxIndex];

			// Note: the CPU version of the algorithm uses only the first occurrency of the max score
			const unsigned a = shared_memory[index + 1];
			const unsigned b = shared_memory[index + 2];
			const unsigned old_a = shared_memory[threadMaxIndex + 1];
			const unsigned old_b = shared_memory[threadMaxIndex + 2];

			if (newMax > oldMax || ((newMax == oldMax) && (a * col + b < old_a * col + old_b)))
				threadMaxIndex = index;
		}

		dev_res[z] = shared_memory[threadMaxIndex];
		unsigned a = shared_memory[threadMaxIndex + 1];
		unsigned b = shared_memory[threadMaxIndex + 2];
	
#ifndef SEPARATE_TRACEBACK
		const unsigned base_path = z * dim_path;

		unsigned n;
		directions dir = directions::invalid;
		for (n = 0; n < dim_path && (dir = dev_entries[matrix_base + a * col + b].dir) != directions::invalid; n++)
		{
			switch (dir)
			{
			case directions::match: case directions::mismatch:	a--;	b--;	break;
			case directions::deletion:							a--;			break;
			case directions::insertion:									b--;	break;
			}
			dev_path[base_path + n] = (char)dir;
		} // end for

		for (; n < dim_path; n++)
			dev_path[base_path + n] = (char)directions::invalid;

#else
		coords[z * 2] = a;
		coords[z * 2 + 1] = b;
#endif
	} // end if
} // ComputeSW


#ifdef SEPARATE_TRACEBACK
__global__ void Traceback(const SW_MatrixEntry* dev_entries,
	const unsigned row, const unsigned col, const int* coords,
	char* dev_path, const unsigned dim_path, const unsigned entries)
{
	const unsigned x = blockIdx.x * blockDim.x + threadIdx.x; // Some threads must not process anything
	if (x < entries)
	{
		const unsigned base_path = x * dim_path;
		const unsigned matrix_base = x * row * col;

		unsigned a = coords[x * 2];
		unsigned b = coords[x * 2 + 1];

		unsigned n;
		directions dir = directions::invalid;
		for (n = 0; n < dim_path && (dir = dev_entries[matrix_base + a * col + b].dir) != directions::invalid; n++)
		{
			switch (dir)
			{
			case directions::match: case directions::mismatch:	a--;	b--;	break;
			case directions::deletion:							a--;			break;
			case directions::insertion:									b--;	break;
			}
			dev_path[base_path + n] = (char)dir;
		} // end for

		for (; n < dim_path; n++)
			dev_path[base_path + n] = (char)directions::invalid;
	}
} // Traceback
#endif