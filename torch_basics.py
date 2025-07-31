import torch


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device}")

    tensor_a = torch.randn(3, 3, device=device, dtype=torch.float32, requires_grad=True)
    tensor_b = torch.randn_like(tensor_a, requires_grad=True)

    print("\ntensor_a:\n", tensor_a)
    print("\ntensor_b:\n", tensor_b)

    sum_tensor = tensor_a + tensor_b
    print("\nSum:\n", sum_tensor)

    matmul_tensor = torch.matmul(tensor_a, tensor_b)
    print("\nMatrix Multiplication:\n", matmul_tensor)

    exp_tensor = torch.exp(tensor_a)
    print("\nexp(tensor_a):\n", exp_tensor)

    flattened = tensor_a.reshape(-1)
    restored  = flattened.view(3, 3)
    print("\nFlatten → Restore shape:", flattened.shape, "→", restored.shape)

    loss = (matmul_tensor + sum_tensor).mean()
    print("\nLoss value:", loss.item())

    loss.backward()

    print("\ntensor_a.grad:\n", tensor_a.grad)
    print("\ntensor_b.grad:\n", tensor_b.grad)


if __name__ == "__main__":
    main()
