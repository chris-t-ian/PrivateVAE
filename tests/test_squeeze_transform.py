import pytest
import torch
from p_vqvae.neural_spline_flow import SqueezeTransform


def test_squeeze_transform_correct_output_shape():
    # Test case 1: Input with spatial dimensions divisible by factor
    squeeze_transform = SqueezeTransform(factor=2)
    input_tensor = torch.randn(4, 1, 8, 8, 8)  # Batch size 4, Channels 1, 8x8x8 spatial dimensions
    output, _ = squeeze_transform(input_tensor)
    expected_shape = (4, 8, 4, 4, 4)  # Channels multiplied by 2^3 (factor^3) and each spatial dimension halved
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

    # Test case 2: Input with different channels and size divisible by factor
    input_tensor = torch.randn(2, 2, 4, 4, 4)  # Batch size 2, Channels 2, 4x4x4 spatial dimensions
    output, _ = squeeze_transform(input_tensor)
    expected_shape = (2, 16, 2, 2, 2)  # Channels multiplied by 2^3 and spatial dimensions divided by 2
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"


def test_squeeze_transform_invalid_input():
    # Test case 3: Input with spatial dimensions not divisible by factor
    squeeze_transform = SqueezeTransform(factor=2)
    input_tensor = torch.randn(4, 1, 7, 8, 8)  # 7 is not divisible by factor
    with pytest.raises(ValueError, match="Input image size not compatible with the factor"):
        squeeze_transform(input_tensor)


def test_squeeze_transform_invalid_dimensions():
    # Test case 4: Input tensor with fewer than 5 dimensions
    squeeze_transform = SqueezeTransform(factor=2)
    input_tensor = torch.randn(4, 1, 8, 8)  # 4D input (Batch, Channels, Height, Width), missing Depth
    with pytest.raises(ValueError, match="Expecting inputs with 5 dimensions"):
        squeeze_transform(input_tensor)


def test_inverse_transform_correct_output_shape():
    # Test case 5: Testing inverse transformation
    squeeze_transform = SqueezeTransform(factor=2)
    input_tensor = torch.randn(4, 1, 8, 8, 8)
    output, _ = squeeze_transform(input_tensor)
    inverse_output, _ = squeeze_transform.inverse(output)
    assert inverse_output.shape == input_tensor.shape, \
        f"Expected shape {input_tensor.shape}, but got {inverse_output.shape}"


def test_inverse_transform_invalid_channels():
    # Test case 6: Channels not divisible by factor^3
    squeeze_transform = SqueezeTransform(factor=2)
    input_tensor = torch.randn(4, 5, 4, 4, 4)  # Channels not divisible by 2^3
    with pytest.raises(ValueError, match="Invalid number of channel dimensions"):
        squeeze_transform.inverse(input_tensor)


if __name__ == "__main__":
    pytest.main()
