import torch

def check_pytorch():
    try:
        # Check PyTorch version
        print(f"PyTorch Version: {torch.__version__}")

        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Running on CPU.")

        # Create a tensor and perform a basic operation
        tensor = torch.tensor([1.0, 2.0, 3.0])
        print(f"Initial Tensor: {tensor}")

        # Perform a basic operation
        tensor = tensor * 2
        print(f"Tensor after multiplication: {tensor}")

        # Create a tensor on GPU (if CUDA is available) and perform an operation
        if torch.cuda.is_available():
            device = torch.device("cuda")
            tensor_gpu = tensor.to(device)
            print(f"Tensor on GPU: {tensor_gpu}")
        else:
            print("CUDA is not available. Skipping GPU tensor operations.")

        print("PyTorch is working properly.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_pytorch()
