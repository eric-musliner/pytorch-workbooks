import torch
import numpy as np


def main():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    scalar = torch.tensor(7)
    print(scalar)
    print(scalar.item())

    vector = torch.tensor([7, 7])
    print(vector)
    print(vector.ndim)
    print(vector.shape)

    MATRIX = torch.tensor([[7, 8], [9, 10]])
    print(MATRIX)
    print(MATRIX.ndim)
    print(MATRIX.shape)

    TENSOR = torch.tensor(
        [
            [
                [1, 2, 4],
                [3, 6, 9],
                [2, 4, 5],
            ],
            [
                [1, 2, 4],
                [3, 6, 9],
                [2, 4, 8],
            ],
        ]
    )
    print(TENSOR)
    print(TENSOR.ndim)
    print(TENSOR.shape)
    print(TENSOR[0])

    # Create random matrix tensor of size (3, 4)
    random_tensor = torch.rand(size=(3, 4))
    print(random_tensor)
    print(random_tensor.ndim)

    # Create a random tensor with similar shape to an image tensor
    random_image_size_tensor = torch.rand(
        size=(224, 224, 3)
    )  # height, width, color channels (R, G, B)
    print(random_image_size_tensor.shape)
    print(random_image_size_tensor.ndim)

    # Zeros and Ones
    # Create tensor of all zeroes
    zeros = torch.zeros(size=random_tensor.size())
    print(zeros)

    print(zeros * random_tensor)

    # Create a tensor of all ones
    ones = torch.ones(size=random_tensor.size())
    print(ones)
    print(ones.dtype)

    # Create a range of tensors and tensors-like
    one_to_ten = torch.arange(start=1, end=11)
    print(one_to_ten)

    # Creating tensors like
    ten_zeros = torch.zeros_like(one_to_ten)
    a = torch.zeros(size=one_to_ten.size())
    print(ten_zeros)
    print(ten_zeros.dtype)
    print(a)
    print(a.dtype)

    # Tensor datatypes
    float_32_tensor = torch.tensor(
        [3.0, 6.0, 9.0], dtype=None, device=device, requires_grad=False
    )
    print(float_32_tensor)
    print(float_32_tensor.dtype)

    float_16_tensor = float_32_tensor.type(torch.half)  # float16
    print(float_16_tensor)
    float_16_tensor = torch.rand(
        size=float_32_tensor.size(), device=device, dtype=torch.half
    )

    print(float_16_tensor * float_32_tensor)

    int_32_tensor = torch.tensor([3, 6, 9], device=device, dtype=torch.int32)
    int_32_tensor * float_32_tensor

    # Getting information from Tensors
    # tensor.dtype, tensor.shape, tensor.device

    print(float_32_tensor.device)

    some_tensor = torch.rand(3, 4)
    print(some_tensor)
    print(some_tensor.shape)

    # Manipulating Tensors (Tensor Operations)
    # Addition, Subtraction, Multiplication (element-wise), Division, Matrix Mutiplication

    tensor = torch.tensor([1, 2, 3])
    print(tensor + 10)
    print(tensor * 10)

    # Pytorch in-built functions
    print(torch.mul(tensor, 10))

    # Matrix Multiplication
    print(tensor, "*", tensor)
    print(f"Equals: {tensor * tensor}")

    print(torch.matmul(tensor, tensor))

    value = 0
    for i in range(len(tensor)):
        value += tensor[i] * tensor[i]
    print(value)

    # Inner dimensions must match
    # COLUMNS OF A = ROWS of B
    a = torch.rand(3, 2)
    print(a)

    # Resulting matrix has the shape of the outer dimensions
    # print(torch.matmul(torch.rand(3, 2), torch.rand(3, 2))) # -> won't work
    print(torch.matmul(torch.rand(2, 3), torch.rand(3, 2)))  # -> shape 2,2
    print(torch.matmul(torch.rand(3, 2), torch.rand(2, 3)))  # -> shape 3,3

    # Shapes for matrix multiplication
    tensor_A = torch.tensor([[1, 2], [3, 4], [5, 6]])
    tensor_B = torch.tensor([[7, 10], [8, 11], [9, 12]])

    # won't work as is because (3, 2) * (3, 2)
    # Fix tensor shape using transpose
    print(tensor_B)
    print(tensor_B.T)
    # torch.mm(tensor_A, tensor_B) # torch.mm is matmul
    print(torch.mm(tensor_A, tensor_B.T))
    print(torch.mm(tensor_A.T, tensor_B))

    # Tensor Aggregation
    # Finding min, max, mean, sum, etc
    x = torch.arange(0, 100, 10)
    print(x)
    print(torch.mean(x.type(torch.float32)))
    print(x.type(torch.float32).mean())
    print(x.sum())

    # Finding positional min/max of Tensors
    print(torch.argmin(x.type(torch.float32)))
    print(torch.argmax(x.type(torch.float32)))
    print(torch.argmin(tensor_B.type(torch.float32)))

    # Reshaping, stacking, squeezing, unsqueezing Tensors
    # Reshaping - reshapes input tensor to a defined shape
    # View - return a view of an input tensor of certain shape but keep the same memory as the original tensor
    # Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
    # Squeeze - remove all `1` dimensions from a tensor
    # Unsqueeze - add a `1` dimension to a target tensor
    # Permute - Return a view of the input with  dimensions permuted (swapped) in a certain way
    x = torch.arange(1.0, 10.0)
    print(x, x.shape)

    # Add an extra dimension
    x_reshaped = x.reshape(1, 9)
    print(x_reshaped)

    # Change the view
    z = x.view(1, 9)
    print(z, z.shape)

    # Changing z changes x because a view shares the memory of the original input
    z[:, 0] = 5
    print(z, x)

    # Stack tensors on top of each other
    x_stacked = torch.stack([x, x, x, x], dim=0)
    print(x_stacked)
    x_stacked = torch.stack([x, x, x, x], dim=1)
    print(x_stacked)

    # Squeeze and Unsqueeze
    # squeeze - removes all single dimensions from a target tensor
    # x = torch.zeros(2, 1, 2, 1, 2)
    # print(x)
    # print(x.size())
    x_squeezed = x.squeeze()
    print(x_squeezed)

    print(x_reshaped.squeeze())

    # Unsqueeze - adds a single dimension to a target tensor at a specific dimenion (dim)
    print(f"Previous target: {x_squeezed}")
    print(f"Previous shape: {x_squeezed.shape}")

    # Add an extra dimension with unsqueeze
    x_unsqueezed = x_squeezed.unsqueeze(dim=0)
    print(f"New Tensor: {x_unsqueezed}")
    print(f"New shape: {x_unsqueezed.shape}")

    # Permute - rearranges dimensions of target tensor in a specified order
    x_original = torch.rand(size=(224, 224, 3))  # [height, width, color channels]

    # Permute original tensor to rearrange the axis (or dim) order
    x_permuted = x_original.permute(2, 0, 1)  # shifts color channel to first index
    print(x_original.shape)
    print(x_permuted.shape)
    print(x_permuted)

    # Selecting data from Tensors (Indexing)
    x = torch.arange(1, 10).reshape(1, 3, 3)
    print(x)
    print(x[0])
    print(x[0][0])
    print(x[0, 0])
    print(x[0][0][0])
    print(x[0, 0, 0])
    print(x[0, 2, 2])

    # You can also use : to select all target dimension
    print(x[:][0])
    print(x[:, 0])

    # Get all values of 0, and 1 dim. But only index 1 of second dimension
    print(x[:, :, 1])

    # Get all values of 0 dimension but only the 1 index value of the 1st and 2nd dimension
    print(x[:, 1, 1])

    # Get the index 0 of 0th and 1st dimension and all values of 2nd
    print(x[0, 0, :])

    # Inex on x to return 3, 6, 9
    print(x[:, :, 2])

    # Tensors and Numpy
    # Data in NumPy, want in PyTorch tensor
    array = np.arange(1, 8)
    tensor = torch.from_numpy(array)
    print(array, tensor.dtype)

    # change value of array
    array = array + 1
    print(array, tensor)

    # Tensor to Numpy
    tensor = torch.ones(7)
    print(tensor)
    numpy_tensor = tensor.numpy()
    print(tensor, numpy_tensor)

    # Change tensor
    tensor = tensor + 1
    print(tensor, numpy_tensor)

    # Reproducibility
    print(torch.rand(3, 3))
    rand_A = torch.rand(3, 4)
    rand_B = torch.rand(3, 4)

    print(rand_A)
    print(rand_B)
    print(rand_A == rand_B)

    # random but reproducible tensors
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    rand_A = torch.rand(3, 4)
    torch.manual_seed(RANDOM_SEED)
    rand_B = torch.rand(3, 4)
    print(rand_A)
    print(rand_B)
    print(rand_A == rand_B)

    # Running tensors and pytorch objects on gpus for faster computations
    print(torch.backends.mps.is_available())

    # Setup device agnostic code
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    tensor = torch.tensor([1, 2, 3])
    print(tensor, tensor.device)

    # Move tensor to gpu device if available
    tensor_on_gpu = tensor.to(device)
    print(tensor_on_gpu, tensor_on_gpu.device)

    # Move tensor back to cpu
    # If tensor on gpu can't transform to cpu
    # print(tensor_on_gpu.numpy())
    tensor_on_cpu = tensor.cpu().numpy()
    print(tensor_on_cpu, tensor_on_cpu.device)

    # Exercises
    tensor = torch.rand(7, 7)
    print(tensor)


if __name__ == "__main__":
    main()
