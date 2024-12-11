from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor, zeros


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # Calculate new dimensions
    new_height = height // kh
    new_width = width // kw

    # Reshape the input tensor
    input = input.contiguous()
    reshaped = input.view(batch, channel, new_height, kh, new_width, kw)
    # Permute dimensions to bring kernel dimensions together
    tiled = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
    # Flatten the kernel dimensions

    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)
    print(tiled.shape)
    print(tiled)
    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform average pooling on a 2D tensor."""
    # Call the tile function to reshape the input tensor
    tiled, new_height, new_width = tile(input, kernel)

    # Compute the mean across the last dimension of the tiled tensor
    # Compute the mean across the last dimension of the tiled tensor
    mean_tiled = tiled.mean(dim=4)

    mean_tiled = mean_tiled.contiguous()

    # Reshape the mean tensor to the desired output shape
    output = mean_tiled.view(input.shape[0], input.shape[1], new_height, new_width)

    # Return the final output tensor
    return output


max_reduce = FastOps.reduce(operators.max, -float("inf"))


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Max the tensor along a dimension"""
        # Convert dim tensor to an integer
        dim = int(dim.item())  # type: ignore
        # Use the max_reduce function to get max values and indices
        max_values = max_reduce(a, dim)  # type: ignore
        # Save the indices and original shape for backward pass
        ctx.save_for_backward(a, dim)

        return max_values

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Max backward"""
        a, dim = ctx.saved_values
        output = argmax(a, dim)
        return grad_output * output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max values and their indices along a specified dimension using the Max class."""
    # Use the Max class to compute the max values
    max_values = Max.apply(input, tensor(dim))
    return max_values


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform max pooling on a 2D tensor."""
    # Call the tile function to reshape the input tensor
    tiled, new_height, new_width = tile(input, kernel)

    # Compute the max across the last dimension of the tiled tensor
    max_tiled = max(tiled, dim=4)

    max_tiled = max_tiled.contiguous()

    # Reshape the max tensor to the desired output shape
    output = max_tiled.view(input.shape[0], input.shape[1], new_height, new_width)

    # Return the final output tensor
    return output


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor along a specified dimension."""
    # Compute the indices of the max values
    max_values = max_reduce(input, dim)
    return input == max_values


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax of a tensor along a specified dimension using the existing max function."""
    # Use the existing max function to compute the maximum values along the specified dimension
    max_values = max(input, dim)
    # Subtract the max for numerical stability
    exp_input = (input - max_values).exp()
    sum_exp = exp_input.sum(dim)
    return exp_input / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax of a tensor along a specified dimension."""
    # Use the softmax function and take the log
    softmax_vals = softmax(input, dim)
    return softmax_vals.log()


def dropout(
    input: Tensor, p: float, training: bool = True, ignore: bool = False
) -> Tensor:
    """Apply dropout to the input tensor based on random noise.

    Args:
    ----
        input: The input tensor.
        p: The probability of dropping a unit.
        training: If True, apply dropout; if False, return the input unchanged.
        ignore: A tensor of the same shape as input, where True indicates elements to ignore during dropout.

    Returns:
    -------
        A tensor with dropout applied.

    """
    if not training or p == 0.0 or ignore:
        return input

    if p == 1.0:
        # Return a tensor of zeros if p is 1.0
        return zeros(input.shape)

    # Create a mask with the same shape as the input
    mask = rand(input.shape) > p

    # if ignore is not None:
    #     # Ensure ignored elements are not dropped
    #     mask = mask | ignore.bool()

    # Scale the mask to maintain the expected value
    scale = 1.0 / (1.0 - p)
    # Apply the mask and scale the output
    return input * mask * scale
