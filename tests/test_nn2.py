from hypothesis import given

import minitorch
from minitorch import Tensor



def test_avgpool2d():
    # Define a sample 4x4 input tensor and manually add a batch dimension
    input_tensor = minitorch.tensor([
        [
            [  # Channel 1
                [1.0, 2.0, 3.0, 0.0],
                [4.0, 5.0, 6.0, 1.0],
                [7.0, 8.0, 9.0, 2.0],
                [3.0, 4.0, 5.0, 6.0]
            ]
        ]
    ]) # This is a 1x1x4x4 tensor (batch x channel x height x width)

    print("Input Tensor:")
    print(input_tensor)

    # Define the kernel size for average pooling
    kernel_size = (2, 2)

    # Apply the minitorch avgpool2d operation
    output_tensor = minitorch.avgpool2d(input_tensor, kernel_size)

    print("\nOutput Tensor:")
    print(output_tensor)

# test_avgpool2d()

def test_max():
    # Create a sample 3D tensor
    input_tensor = minitorch.tensor([
        [
            [1.0, 3.0, 2.0],
            [4.0, 6.0, 5.0]
        ],
        [
            [7.0, 9.0, 8.0],
            [10.0, 12.0, 11.0]
        ]
    ])  # Shape: (2, 2, 3)


    # Call the max function from nn.py
    max_values = minitorch.nn.max(input_tensor, dim=2)

    print("Max Values:")
    print(max_values)


# Run the test
test_max()
