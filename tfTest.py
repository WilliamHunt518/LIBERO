import torch

if __name__ == '__main__':

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        # Get the CUDA device name
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA Device Name: {device_name}")

        # Perform a simple tensor operation on the GPU
        try:
            # Create two random tensors
            x = torch.rand(3, 3).cuda()
            y = torch.rand(3, 3).cuda()

            # Perform matrix multiplication
            z = torch.matmul(x, y)

            print("Tensor operation successful. Result:")
            print(z)
        except Exception as e:
            print(f"An error occurred during tensor operations: {e}")
    else:
        print("CUDA is not available. The script will not perform GPU operations.")
