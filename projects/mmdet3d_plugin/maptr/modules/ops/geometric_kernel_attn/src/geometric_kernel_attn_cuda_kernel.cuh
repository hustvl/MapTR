#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>
#include <algorithm>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

__device__ int clip(int n, int lower, int upper) {
  n = n >= lower ? n : lower;
  return n < upper ? n : upper;
}

template <typename scalar_t>
__device__ scalar_t multi_scale_kernel_attn_sampling(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const int &h,
    const int &w, const int &m, const int &c) {
  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;

  const int base_ptr = m * channels + c;
  const int h_ptr_offset = h_stride * h;
  const int w_ptr_offset = w_stride * w;
  scalar_t val = bottom_data[base_ptr + h_ptr_offset + w_ptr_offset];

  return val;
}

template <typename scalar_t>
__device__ void multiscale_kernel_attn_sampling_backward(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const int &h,
    const int &w, const int &m, const int &c, const scalar_t &top_grad,
    const scalar_t &attn_weight, scalar_t *&grad_value,  scalar_t *grad_attn_weight) {

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_ptr_offset = h_stride * h;
  const int w_ptr_offset = w_stride * w;
  const int base_ptr = m * channels + c;
  const scalar_t top_grad_value = top_grad * attn_weight;
  // scalar_t grad_h_weight = 0, grad_w_weight = 0;

  const int ptr = base_ptr + h_ptr_offset + w_ptr_offset;
  scalar_t val = bottom_data[ptr];
  atomicAdd(grad_value + ptr, top_grad_value);
  *grad_attn_weight = top_grad * val;
}


template <typename scalar_t>
__global__ void multiscale_kernel_attn_forward_gpu_kernel(
    const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const int64_t *data_sampling_loc,
    const scalar_t *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    scalar_t *data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const int loc_w = data_sampling_loc[data_loc_w_ptr];
        const int loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];
        const int loc_h_ = clip(loc_h, 0, spatial_h-1);
        const int loc_w_ = clip(loc_w, 0, spatial_w-1);
        col += multi_scale_kernel_attn_sampling(data_value_ptr, spatial_h, spatial_w, num_heads, 
                                                channels, loc_h_, loc_w_, m_col, c_col) * weight;

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void multiscale_kernel_attn_backward_gpu_kernel_shm_blocksize_aware_reduce_v2(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const int64_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    // grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    // const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const int loc_w = data_sampling_loc[data_loc_w_ptr];
        const int loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];
        *(cache_grad_attn_weight+threadIdx.x)=0;
        const int loc_h_ = clip(loc_h, 0, spatial_h-1);
        const int loc_w_ = clip(loc_w, 0, spatial_w-1);
        multiscale_kernel_attn_sampling_backward(
          data_value_ptr, spatial_h, spatial_w, num_heads, channels, loc_h_, loc_w_, m_col, c_col,
          top_grad, weight, grad_value_ptr, cache_grad_attn_weight+threadIdx.x);
        __syncthreads();

        for (unsigned int s=blockSize/2; s>0; s>>=1)
        {
          if (tid < s) {
            // const unsigned int xid1 = tid << 1;
            //const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
          }
          __syncthreads();
        }

        if (tid == 0)
        { 
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
      }
    }
  }
}


template <typename scalar_t>
__global__ void multiscale_kernel_attn_backward_gpu_kernel_shm_reduce_v2(
  const int n,
  const scalar_t *grad_col,
  const scalar_t *data_value,
  const int64_t *data_spatial_shapes,
  const int64_t *data_level_start_index, 
  const int64_t *data_sampling_loc,
  const scalar_t *data_attn_weight,
  const int batch_size, 
  const int spatial_size, 
  const int num_heads,
  const int channels, 
  const int num_levels,
  const int num_query,
  const int num_point,
  scalar_t *grad_value,
  scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    // grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    // const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const int loc_w = data_sampling_loc[data_loc_w_ptr];
        const int loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];
        *(cache_grad_attn_weight+threadIdx.x)=0;
        const int loc_h_ = clip(loc_h, 0, spatial_h-1);
        const int loc_w_ = clip(loc_w, 0, spatial_w-1);
        multiscale_kernel_attn_sampling_backward(
          data_value_ptr, spatial_h, spatial_w, num_heads, channels, loc_h_, loc_w_, m_col, c_col,
          top_grad, weight, grad_value_ptr, cache_grad_attn_weight+threadIdx.x);
        __syncthreads();
        
        for (unsigned int s=blockDim.x/2, spre=blockDim.x; s>0; s>>=1, spre>>=1)
        {
          if (tid < s) {
            // const unsigned int xid1 = tid << 1;
            // const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            if (tid + (s << 1) < spre)
            {
              cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + (s << 1)];

            } 
          }
          __syncthreads();
        }

        if (tid == 0)
        {
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
      }
    }
  }
}


template <typename scalar_t>
void multiscale_kernel_attn_forward_cuda(cudaStream_t stream,
                              const scalar_t* data_value,
                              const int64_t* data_spatial_shapes, 
                              const int64_t* data_level_start_index, 
                              const int64_t* data_sampling_loc,
                              const scalar_t* data_attn_weight,
                              const int batch_size,
                              const int spatial_size, 
                              const int num_heads, 
                              const int channels, 
                              const int num_levels, 
                              const int num_query,
                              const int num_point,
                              scalar_t* data_col)
{
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;
  multiscale_kernel_attn_forward_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
          0, stream>>>(
      num_kernels, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight, 
      batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in multiscale_kernel_attn_forward_cuda: %s\n", cudaGetErrorString(err));
  }

}


template <typename scalar_t>
void multiscale_kernel_attn_backward_cuda(cudaStream_t stream,
                                          const scalar_t* grad_col,
                                          const scalar_t* data_value,
                                          const int64_t * data_spatial_shapes,
                                          const int64_t * data_level_start_index,
                                          const int64_t * data_sampling_loc,
                                          const scalar_t * data_attn_weight,
                                          const int batch_size, 
                                          const int spatial_size, 
                                          const int num_heads,
                                          const int channels, 
                                          const int num_levels,
                                          const int num_query,
                                          const int num_point, 
                                          scalar_t* grad_value,
                                          scalar_t* grad_attn_weight)
{
  const int num_threads = (channels > CUDA_NUM_THREADS)?CUDA_NUM_THREADS:channels;
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  switch(channels) {
    case 128:
    multiscale_kernel_attn_backward_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_attn_weight);
      break;
    case 256:
    multiscale_kernel_attn_backward_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_attn_weight);
      break;
    case 512:
    multiscale_kernel_attn_backward_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_attn_weight);
      break;
    case 1024:
    multiscale_kernel_attn_backward_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_attn_weight);
      break;
    default:
      multiscale_kernel_attn_backward_gpu_kernel_shm_reduce_v2<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, num_threads*3*sizeof(scalar_t), stream>>>(
                    num_kernels, 
                    grad_col,
                    data_value,
                    data_spatial_shapes,
                    data_level_start_index, 
                    data_sampling_loc,
                    data_attn_weight,
                    batch_size, 
                    spatial_size, 
                    num_heads,
                    channels, 
                    num_levels,
                    num_query,
                    num_point,
                    grad_value,
                    grad_attn_weight);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in multiscale_kernel_attn_backward_cuda: %s\n", cudaGetErrorString(err));
  }

}
