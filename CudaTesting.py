# Through this script we will going to learn Cuda(Compute Unified Device Architecture) in python.
# Requirements

from cuda_available import *
from numba import jit, vectorize, cuda
import numpy as np
import math
import timeit
import webbrowser

""" 
    In simpler words CUDA is a software as well as hardware component developed by Nvidia. It is use to speed up the
    programming by running programs on GPUs instead on CPU specifically mathematical programs. It is a platform as well as
    present in GPU as hardware too.CUDA leverages the massive parallel computing power of NVIDIA GPUs, allowing for significant
    speedups in computations that can be parallelized.
"""

# Lets first check is our device compatible for cuda or not by running small script.

# cnt = getCudaDeviceCount()
# ver = cudaDriverVersion()

# if cnt>=1:
#     print(f"Success you have {cnt} cuda device present in your GPU and its version is {ver}")
#     info = input("Want to get more info about your cuda device? Y/N-->")
#     if info == "y" or info == "Y":
#         for idx in range(cnt):
#             full_info = CudaDeviceInfo(idx)
#             print(f"UUID: {full_info.uuid}")
#             print(f"Name: {full_info.name}")
#             print(f"ComputeCapability: {full_info.computeCapability}")
#             print(f"TotalGlobalVmen: {full_info.totalGlobalVmem}")
#             print(f"PciID: {full_info.pciId}")
#             print(f"UsingTccDriver: {full_info.isTccDriver}")
# else:
#     print("Failed! Check if your device have a nvidia graphics card.")
# run above script by de-commenting it

"""
    To Utilize cuda in python programming we use a library called numba resemblance with numpy.In simple terms numba
    is a just in time function compiler that is numerically focused too much to handle?? don't worry let's break it
    1 --> Function compiler:
           Numba make changes in functional data types as this data types are not suitable for gpu programming hence it 
           convert them into simple data structures so as to make overall function fast.
    2 --> Just in time
          It's pretty to understand by just in time in programming no? So by just in time means that Numba make necessary 
          changes in function when the function is interpreted.
    3 --> Numerically-focused
          At the current stage Numba is able to handle some numeric data types only which includes int,float and comples
          it can work every few string operation hence it is referred as numerically focused. 
"""

# Let's first optimize our code at cpu level for the entrance sake than we move to gpu programming

"""
    The Numba compiler is basically implemented using function decorators to Python function 
    Decorators are function modifiers that transform the Python functions they decorate, 
    using a very simple syntax. Here we will use Numba's CPU compilation decorator @jit 
"""

# @jit
# def hypot(x:float,y:float)->float:
#     x = abs(x)
#     y = abs(y)
#     t = min(x,y)
#     x = max(x,y)
#     t = t/x
#     return x * math.sqrt(1+t*t)
# print(timeit.timeit("hypot(3.0,4.0)",setup="from __main__ import hypot")) # Measures the execution time of jit function # fast
# print(timeit.timeit("hypot.py_func(3.0,4.0)",setup="from __main__ import hypot")) # Faster
# print(timeit.timeit("math.hypot(3.0,4.0)",setup="from __main__ import math")) # fastest

""" 
    Here our @jit decorator works properly but not gives expected results because of 
    Numba does introduce some overhead to each function call that is larger than 
    the function call overhead of Python itself.
    Let's see some simple example to overcome such situations.
"""

# def orifact(x):
#     if x == 1:
#         return 1
#     return x*fact(x-1)
# @jit
# def fact(x: int) -> int:
#     if x == 1:
#         return 1
#     return x * fact(x - 1)
#
# orifact(20)
# fact(20)
# print(timeit.timeit("fact(20)", setup="from __main__ import fact"))  # this is gpu function
#
# # Now our won fact fucntion with same logic
# print(timeit.timeit("orifact(20)", setup="from __main__ import orifact"))

# @jit
# def fact(x: int) -> int:
#     if x == 1:
#         return 1
#     return x * fact(x - 1)
#
# Warm-up call to exclude the compilation time from the timing
# fact(20)
#
# # Measure the execution time of the JIT-compiled fact function
# jit_time = timeit.timeit("fact(20)", setup="from __main__ import fact", number=10000) #Fastest
# print(f"JIT-compiled function time: {jit_time}")
#
# Measure the execution time of the pure Python fact function
# py_time = timeit.timeit("fact.py_func(20)", setup="from __main__ import fact", number=10000) #Fast
# print(f"Pure Python function time: {py_time}")
#
# print("In built math fucntion >>",timeit.timeit("math.factorial(20)",setup ="from __main__ import math",number=10000)) # faster

#######################################################################################################################################################################################

# Now we use Numba compiler for its actual purpose that is for GPUs with Numpy Universal Functions(ufuncs)

"""
Knowledge>>
GPUs are designed to mainly focus on data parallelism.Buy why data parallelism? because to display thousands of pixels at very 
fast rate GPU does same operation on a element many times for an instance like we all know that pixels on our screen are stored
in the form of matrix in memory so GPU might need matrix multiplication to render an image hence GPU perform matrix multiplication
many times on number of matrix at once that what we called data parallelism. 
"""

"""
Why use Numpy Universal functions?
--> Numpy ufuncs perform same opration on multiple data points which is naturally data parallelism hence it is fit for GPU programming.
"""
# a = np.array([1, 2, 3, 4, 5])
# b = np.array([10, 20, 30, 40, 50])  # b= 10 also works the same due to broadcasting feature of numpy
# print(np.add(a, b))  # here add is Universal functions
# print(np.add(100, a))
# c = np.arange(4*4).reshape((4,4))
# print(c+10)
# It is suggest here that you have some basic knowledge of numpy if not than read this basic manual https://numpy.org/doc/stable/user/quickstart.html

"""
    But wait if there are numpy ufuncs than why we need numba? 
--> Numba has decorator called vectorized those who know numpy vectorize are will get ease to understand numba's vectorize
    and numpy can only work with CPUs but numba can  
"""

# Making ufuncs for Gpu
# This is a CPU optimized function

# Function to just optimize cpu a little bit
# @vectorize
# def add(a, b):
#     return a + b

"""     
    This function use gpu resources and does calculation in gpu
    But when you run below mentioned program you will get warning that the program is too small to occupy gpu.  
"""
# @vectorize(['int64(int64,int64)'], target='cuda')
# def add_ufunc(a, b):
#     return a + b
# a = np.arange(10)
# b = np.arange(10)
# print(np.add(a, b))
# print(add_ufunc(a, b))
# print(cuda.gpus)
"""
 1.why this warning??
 --> Cause 1 Our input is too small
        The GPU achieves performance through parallelism, operating on thousands of values at once.
        Our test inputs have only 10 integers, respectively.  We need a much larger array to 
        even keep the GPU busy.
         
     Cause 2 Our calculation is too simple
        Sending a calculation to the GPU involves quite a bit of overhead compared to calling a function on the CPU.
        If our calculation does not involve enough math operations (often called "arithmetic intensity") 
        then the GPU will spend most of its time waiting for data to move around.
     
     Cause 3 Here we are coping the data to and from the GPU   
        In small program like this time to transfer data from cpu to gpu can take more time tha gpu itself take
        to run the program, Similarly when the operation are done in gpu sending result back to cpu take time too.
        
        Example Scenario
            Inefficient Use:
                Transfer data to GPU.
                Perform one simple calculation.
                Transfer data back to CPU.
                Repeat these steps for each calculation.
            Efficient Use:
                Transfer data to GPU.
                Perform multiple calculations on the GPU.
                Transfer the final result back to CPU once all calculations are done.
        By minimizing the number of times we transfer data between the CPU and GPU, 
        we can significantly improve performance.
    
    Cause 4 Our data types are larger than necessary
        Our example uses `int64` when we probably don't need it.Instead we can use float32 which is amiable for 
        the function and also runs faster on GPU. 
"""

# Evaluate the Gaussian a million times!
# SQRT_2PI = np.float32((2*math.pi)**0.5)  # Precompute this constant as a float32.  Numba will inline it at compile time.

"""This is a GPU optimize function.If we remove vectorize decorator than it will become a CPU function"""
# x = np.random.uniform(-3, 3, size=1000000).astype(np.float32)

# @vectorize(['float32(float32, float32, float32)'], target='cuda')
# def gaussian_pdf(x, mean, sigma):
#     return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)
# mean = np.float32(0.0)
# sigma = np.float32(1.0)

# print(gaussian_pdf(x, 0.0, 1.0))

####################################################################################################################################################################################
# Managing GPU Memory
"""
    So far we have used NumPy arrays on the CPU as inputs and outputs to our GPU functions. As a convenience, Numba has been
    automatically transferring this data to the GPU for us so that it can be operated on by the GPU. With this implicit data 
    transfer Numba, acting conservatively, will automatically transfer the data back to the CPU after processing. As you can
    imagine, this is a very time intensive operation.
    
    The CUDA best practice indicates that
    Minimize data transfer between the host and the device(GPU), even if it means running some kernels on the device that do not 
    show performance gains when compared with running them on the host CPU.So its clear that it is infeasible to use Numba to 
    transfer data to GPU.
    
    To achieve this we create CUDA Device Arrays and pass them to our GPU functions. CUDA device arrays, also known as GPU arrays
    or device arrays, are special types of arrays allocated on the GPU's memory. These arrays are used for performing computations
    on the GPU, allowing for parallel processing of large datasets. CUDA device arrays are typically managed using libraries such 
    as Numba or CuPy, which provide a high-level interface for GPU programming in Python.
"""
# Example 1
# n = 100000
# cpu_array = np.arange(n).astype(np.float32)
# gpu_array = cuda.to_device(cpu_array)
# print(gpu_array.shape)

# Example 2
# @vectorize(['float32(float32,float32)'],target='cuda')
# def add_ufunc(x,y):
#     return x + y
# n = 100000
# x = np.arange(n).astype(np.float32)
# y = 2 * x
# print(timeit.timeit('add_ufunc(x,y)',setup="from CudaTesting import add_ufunc, x, y",number=100))

"""
    In this approach first we create array in cpu than transfer it to gpu than again result fro gpu transferred to cpu
    Gpu has special memory for storing array to access this area we can use Cuda device array.    
"""

# gpu_x = cuda.to_device(x)
# gpu_y = cuda.to_device(y)
# print(timeit.timeit("add_ufunc(gpu_x,gpu_y)",setup="from CudaTesting import add_ufunc,gpu_x,gpu_y"))

"""
    Now there is one more overhead is here which is while returning the value.So we have to store the returned value
    # also in gpu's memory. To achieve this 
"""

# out_device = cuda.device_array(shape=(n,),dtype=np.float32)
# This will create a array in gpu memory that has size of n and data type of float 32 bit
# add_func(gpu_x,gpu_y,out = out_device)
# This will store the out value in out device array which will be present only in gpu memory to access it locally we can do
# out_host = out_device.copy_to_host()

###########################################################################################################################################################################################

# Custom Cuda Kernels in Python with Numba

## Introduction to CUDA kernels
"""
    When programming in CUDA, developers write functions for the GPU called kernels, which are executed, or in CUDA parlance,
    launched, on the GPU's many cores in parallel threads. When kernels are launched, programmers use a special syntax, 
    called an execution configuration (also called a launch configuration) to describe the parallel execution's configuration.
    
    To get a dipper understanding of execution of custom cuda kernels watch below mentioned slide by nvidia
"""
# url = "https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-02-V1/AC_CUDA_Python_1.pptx"
# webbrowser.open(url)

"""
Terms to be remembered while building kernels
 for example here we take a kernel signature --> do_work[2,4](x,y)
    1) Threads -> Small piece of process by breaking down a large process.Cuda can run thousands of threads in parallel
    2) Block -> Collection of thread is called block.
    3) Grid -> Collection of blocks are called Grid.
    4) Kernels -> Nothing but functions in GPU  
    5) Execution -> Launched ex:function executed will be kernel launched
    6) Execution Configuration -> Defines number of block per grid and threads per block in our case there 
                                  are 2 blocks in 1 grid and 4 threads in one block i.e given as [2,4] in signature.
                                  
        // Pre-define variables name in kernel are as follows.
         
    7) gridDim.x -> Shorthand of grid dimensions which gives the numbers of blocks in grid which is 2 here. 
    8) blockIdx.x -> It is the index of current block in the grid that can lie in the range of (0,1) here.
    9) blockDim.x -> Dimension of the block in grid which 4 here.
    10) threadIdx.x -> Id of a thread in particular block that lies in between [0,1,2,3] here.
    By using the formula threadIdx.x + blockId.x * blockDim.x we can  find the thread's unique index within grid.
    
    Let's see a simple example of addition  
"""
# addition program
# @cuda.jit
# def addKernel(x, y, out):
#     idx = cuda.grid(1)  # Making a one dimensional grid
#     out[idx] = x[idx] + y[idx]  # 1d out grid stores addition of 1d x grid and 1d y grid.
#
# n = 16384
# x = np.arange(n).astype(np.int32)
# y = np.ones_like(x)
#
# gpu_x = cuda.to_device(x)
# gpu_y = cuda.to_device(y)
# out_device = cuda.device_array(shape=(n),dtype=np.int32)
#
# threads_per_block = 128
# block_per_grid = 129
#
#
# addKernel[block_per_grid,threads_per_block](gpu_x,gpu_y,out_device)
# cuda.synchronize()
# print(out_device.copy_to_host())
#
# Square program
# @cuda.jit
# def square_device(a,out):
#     idx = cuda.grid(1)
#     out[idx] = a[idx]**2
# n = 4096
# a = np.arange(n)
# d_a = cuda.to_device(a)
# d_out = cuda.device_array(shape=(n,),dtype=np.float32) # TODO: make d_out a device array
#
# blocks = 128
# threads = 32
#
# square_device[blocks,threads](d_a,d_out)
# print(d_out.copy_to_host())

"""
Working on Largest Datasets with Grid Stride Loops
    Run the following slides that gives you a high level overview of a technique called a grid stride loop which will create
    flexible kernels where each thread is able to work on more than one data element, an essential technique for large datasets.
    Execute the cell to load the slides.
"""
# url = 'https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-02-V1/AC_CUDA_Python_2.pptx'
# webbrowser.open(url)

# Addition with grid strides
"""
    Let's refactor the add_kernel above to utilize a grid stride loop so that we can launch it to work on larger data sets flexibly
    while incurring the benefits of global memory coalescing, which allows parallel threads to access memory in contiguous chunks,a 
    scenario which the GPU can leverage to reduce the total number of memory operations:
"""
# @cuda.jit
# def addition_with_strides(a,b,c):
#     start = cuda.grid(1)
#     stride = cuda.gridsize(1)  # could be treats as jumps in grid
#     for i in range(start,a.shape[0],stride):
#         c[i] = a[i] + b[i]
#
# a = np.arange(100000).astype(np.int32)
# b = np.ones_like(a)
# c = cuda.device_array(shape=(n),dtype=np.int32)
#
# gpu_a = cuda.to_device(a)
# gpu_b = cuda.to_device(b)
# addition_with_strides[30,128](gpu_a,gpu_b,c)
# print(c.copy_to_host())

"""
Atomic Operations and Avoiding Race Conditions
    CUDA, like many general purpose parallel execution frameworks, makes it possible to have race conditions in your code. 
    A race condition in CUDA arises when threads read to or write from a memory location that might be modified by another
    independent thread. Generally speaking, you need to worry about:

    read-after-write hazards: One thread is reading a memory location at the same time another thread might be writing to it.
    write-after-write hazards: Two threads are writing to the same memory location, and only one write will be visible when 
    the kernel is complete.
    
     A common strategy to avoid both of these hazards is to organize your CUDA kernel algorithm such that each thread has
     exclusive responsibility for unique subsets of output array elements, and/or to never use the same array for both input
     and output in a single kernel call
     
     However, there are many cases where different threads need to combine results. Consider something very simple, like: 
     "every thread increments a global counter." Implementing this in your kernel requires each thread to:
     Read the current value of a global counter.
     Compute counter + 1.
     Write that value back to global memory.
     However, there is no guarantee that another thread has not changed the global counter between steps 1 and 3. To resolve 
     this problem, CUDA provides atomic operations which will read, modify and update a memory location in one, indivisible step.
     Numba supports several of these functions, described here.
    
    Let's make our thread counter kernel:
"""
# Atomic counter operation
# @cuda.jit
# def thread_counter_safe(global_counter):
#     cuda.atomic.add(global_counter,0,1)
#
# global_counter = cuda.to_device(np.array([0],dtype=np.int32))
# thread_counter_safe[64,64](global_counter)
# print(global_counter.copy_to_host())

####################################################################################################################################################################################################

# Effective use of memory and subsystems

#  Uncoalesced Memory Access Hurts Performance
""" 
    Before we learn the details about what coalesced memory access is, let's run matrix add program to observe the performance
    implications for a seemingly trivial change to the data access pattern within a kernel.
"""

"""
n = 1024 * 1024  # nearly equal to 1M
threads = 1024
blocks = 1024

strides = 16  # jumps in grid

a = np.ones(n).astype(np.float32)
b = a.copy().astype(np.float32)
out = np.zeros(n).astype(np.float32)

gpu_a = cuda.to_device(a)
gpu_b = cuda.to_device(b)
gpu_out = cuda.to_device(out)

is_coalesced = False

@cuda.jit
def add_experiment(a, b, out, stride, coalesced):
    i = cuda.grid(1)
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < a.size:  # Ensure i is within bounds
        if coalesced:
            out[i] = a[i] + b[i]
        else:
            out[i] = a[stride * i] + b[stride * i]

# Make sure to pass the correct variables into the timeit setup
setup = "from __main__ import add_experiment, blocks, threads, gpu_a, gpu_b, gpu_out, strides,is_coalesced"
# Measure the execution time by changing the value of is_coalesced variable to True than False
print(
    timeit.timeit('add_experiment[blocks, threads](gpu_a, gpu_b, gpu_out, strides,True)', setup=setup, number=100)
)
"""
"""
Let's dive into memory coalescing 
"""
# url = 'https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-02-V1/coalescing-v3.pptx'
# webbrowser.open(url)  # run this to see prestation on memory coalcsing by nvidia

"""
Now lets run row_sum and  col_sum program to 
"""
# n = 16384
# threads = 256
# blocks = int(n / threads)
# a = np.ones(n*n).reshape(n,n).astype(np.float32)
# a[2] = 9
# sums = np.zeros(n).astype(np.float32)
# gpu_a=cuda.to_device(a)
# gpu_sums = cuda.to_device(sums)
#
# @cuda.jit
# def row_sum(a,sums,n):
#     idx = cuda.grid(1)
#     sum = 0.0
#     if idx<a.shape[0]:
#         for i in range(n):
#             sum += a[i,idx]
#     sums[idx] = sum
#
# def col_sum(a,sums,n):
#     idx = cuda.grid(1)
#     sum =0.0
#     if idx < a.size[0]:
#         for i in range(n):
#             sum += a[idx,i]
#     sums[idx] = sum
#
# print(timeit.timeit("row_sum[blocks,threads](gpu_a,gpu_sums,n)",setup="from __main__ import row_sum,n,threads,blocks,gpu_a,gpu_sums"))
#
# print(timeit.timeit("col_sum[blocks,threads](gpu_a,gpu_sums,n)",setup="from __main__ import row_sum,n,threads,blocks,gpu_a,gpu_sums"))

"""
    2 and 3 Dimensional Blocks and Grids
    Both grids and blocks can be configured to contain a 2 or 3 dimensional collection of blocks or threads, respectively.
    This is done mostly as a matter of convenience for programmers who often work with 2 or 3 dimensional datasets. 
    Here is a very trivial example to highlight the syntax. You may need to read both the kernel definition and its launch 
    before the concept makes sense. 
"""
# a = np.zeros((4,4))
# out = cuda.to_device(a)
# blocks = (2,2)
# threads = (2,2)
# @cuda.jit
# def array_index(a):
#     x,y = cuda.grid(2)
#     a[x][y] = x + y/10
# array_index[blocks,threads](out)
# print(out.copy_to_host())

"""
Shared memory
    We will now discuss how to utilize a region of on-chip device memory called shared memory. Shared memory is a programmer
    defined cache of limited size that depends on the GPU being used and is shared between all threads in a block. It is a 
    scarce resource, cannot be accessed by threads outside of the block where it was allocated, and does not persist after
    a kernel finishes executing. Shared memory however has a much higher bandwidth than global memory and can be used to 
    great effect in many kernels, especially to optimize performance.
    
    When declaring shared memory, you provide the shape of the shared array, as well as its type, using a Numba type. 
    The shape of the array must be a constant value, and therefore, you cannot use arguments passed into the function, 
    or, provided variables like numba.cuda.blockDim.x, or the calculated values of cuda.griddim. Here is a convoluted 
    example to demonstrate the syntax with comments pointing out the movement from host memory to global device memory, 
    to shared memory, back to global device memory, and finally back to host memory.
"""

# @cuda.jit
# def swap_with_shared(vector, swapped):
#     # Allocate a 4 element vector containing int32 values in shared memory.
#     temp = cuda.shared.array(4, dtype=np.int32)
#     idx = cuda.grid(1)
#
#     # Move an element from global memory into shared memory
#     temp[idx] = vector[idx]
#
#     # cuda.syncthreads will force all threads in the block to synchronize here, which is necessary because...
#     cuda.syncthreads()
#     # ...the following operation is reading an element written to shared memory by another thread.
#
#     # Move an element from shared memory back into global memory
#     swapped[idx] = temp[3 - cuda.threadIdx.x]  # swap elements # It does swapped[0] = temp[3] ,swapped[1] = temp[2] so on...
#
# vector = np.arange(4).astype(np.int32)
# swapped = np.zeros_like(vector)
#
# # Move host memory to device (global) memory
# d_vector = cuda.to_device(vector)
# d_swapped = cuda.to_device(swapped)
# swap_with_shared[1,4](d_vector, d_swapped)
# print(d_swapped.copy_to_host())

""" 
Presentation: Shared Memory for Memory Coalescing
"""
# url = 'https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-02-V1/shared_coalescing.pptx'
# webbrowser.open(url)

"""
Coalesced Reads and Writes for matrix transpose using shared memory
"""

# n = 4096*4096 # 16M
# threads_per_block = (32, 32)
# blocks = (128, 128)
#
# a = np.arange(n).reshape((4096,4096)).astype(np.float32)
# transposed = np.zeros_like(a).astype(np.float32)
#
# d_a = cuda.to_device(a)
# d_transposed = cuda.to_device(transposed)
#
# @cuda.jit
# def tile_transpose(a, transposed):
#     # `tile_transpose` assumes it is launched with a 32x32 block dimension,
#     # and that `a` is a multiple of these dimensions.
#
#     # 1) Create 32x32 shared memory array.
#     tile = cuda.shared.array((32, 32), np.float32)
#
#
#     x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#     y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
#
#     tile[cuda.threadIdx.y, cuda.threadIdx.x] = a[y, x]
#
#     # Wait for all threads in the block to finish updating shared memory.
#     cuda.syncthreads()
#
#     t_x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.x
#     t_y = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.y
#
#     transposed[t_y, t_x] = tile[cuda.threadIdx.x, cuda.threadIdx.y]
#
# tile_transpose[blocks,threads_per_block](d_a,d_transposed)
# cuda.synchronize()
# print(d_transposed.copy_to_host())

"""
Presentation: Memory Bank Conflicts
"""
# url = 'https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-02-V1/bank_conflicts.pptx'
# webbrowser.open(url)
"""
# Now it time to you explore the accelerated computing with CUDA Python at https://docs.nvidia.com/cuda/ 
"""