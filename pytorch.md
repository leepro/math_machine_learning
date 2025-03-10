| **Category**             | **Functions**                                                                                                   | **Example**                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **Tensor Creation**      | `torch.tensor`                                                                                                 | `torch.tensor([1, 2, 3])` → `tensor([1, 2, 3])`                                             |
|                          | `torch.zeros`                                                                                                  | `torch.zeros(3)` → `tensor([0., 0., 0.])`                                                  |
|                          | `torch.ones`                                                                                                   | `torch.ones(2)` → `tensor([1., 1.])`                                                       |
|                          | `torch.arange`                                                                                                 | `torch.arange(0, 5, 2)` → `tensor([0, 2, 4])`                                              |
|                          | `torch.linspace`                                                                                               | `torch.linspace(0, 1, 3)` → `tensor([0., 0.5, 1.])`                                        |
|                          | `torch.eye`                                                                                                    | `torch.eye(2)` → `tensor([[1., 0.], [0., 1.]])`                                            |
|                          | `torch.full`                                                                                                   | `torch.full((3,), 7)` → `tensor([7, 7, 7])`                                                |
|                          | `torch.empty`                                                                                                  | `torch.empty(2)` → `tensor([random values])` (uninitialized)                                |
|                          | `torch.diag`                                                                                                   | `torch.diag(torch.tensor([1, 2]))` → `tensor([[1, 0], [0, 2]])`                            |
| **Tensor Manipulation**  | `torch.reshape`                                                                                                | `torch.reshape(torch.tensor([1, 2, 3, 4]), (2, 2))` → `tensor([[1, 2], [3, 4]])`           |
|                          | `torch.split`                                                                                                  | `torch.split(torch.tensor([1, 2, 3, 4]), 2)` → `(tensor([1, 2]), tensor([3, 4]))`          |
|                          | `torch.vstack`                                                                                                 | `torch.vstack((torch.tensor([1, 2]), torch.tensor([3, 4])))` → `tensor([[1, 2], [3, 4]])`  |
|                          | `torch.hstack`                                                                                                 | `torch.hstack((torch.tensor([1, 2]), torch.tensor([3, 4])))` → `tensor([1, 2, 3, 4])`      |
|                          | `torch.tile`                                                                                                   | `torch.tile(torch.tensor([1, 2]), (2,))` → `tensor([1, 2, 1, 2])`                          |
|                          | `torch.tensor_split`                                                                                           | `torch.tensor_split(torch.tensor([1, 2, 3, 4]), 2)` → `(tensor([1, 2]), tensor([3, 4]))`   |
|                          | `torch.cat`                                                                                                    | `torch.cat((torch.tensor([1, 2]), torch.tensor([3, 4])))` → `tensor([1, 2, 3, 4])`         |
|                          | `torch.flip`                                                                                                   | `torch.flip(torch.tensor([1, 2, 3]), [0])` → `tensor([3, 2, 1])`                           |
|                          | `torch.transpose`                                                                                              | `torch.transpose(torch.tensor([[1, 2], [3, 4]]), 0, 1)` → `tensor([[1, 3], [2, 4]])`       |
|                          | `torch.squeeze`                                                                                                | `torch.squeeze(torch.tensor([[1], [2]]))` → `tensor([1, 2])`                                |
| **Mathematical Operations** | `torch.matmul`                                                                                              | `torch.matmul(torch.tensor([1, 2]), torch.tensor([3, 4]))` → `tensor(11)` (1×3 + 2×4)      |
|                          | `torch.exp`                                                                                                    | `torch.exp(torch.tensor(1.))` → `tensor(2.718…)`                                            |
|                          | `torch.sqrt`                                                                                                   | `torch.sqrt(torch.tensor(4.))` → `tensor(2.)`                                               |
|                          | `torch.max`                                                                                                    | `torch.max(torch.tensor([1, 3, 2]))` → `tensor(3)`                                         |
|                          | `torch.min`                                                                                                    | `torch.min(torch.tensor([1, 3, 2]))` → `tensor(1)`                                         |
|                          | `torch.sum`                                                                                                    | `torch.sum(torch.tensor([1, 2, 3]))` → `tensor(6)`                                         |
|                          | `torch.prod`                                                                                                   | `torch.prod(torch.tensor([2, 3]))` → `tensor(6)`                                           |
|                          | `torch.abs`                                                                                                    | `torch.abs(torch.tensor(-5))` → `tensor(5)`                                                |
|                          | `torch.log`                                                                                                    | `torch.log(torch.tensor(2.718))` → `tensor(~1.0)`                                          |
|                          | `torch.sin`                                                                                                    | `torch.sin(torch.tensor(0.))` → `tensor(0.)`                                               |
|                          | `torch.cos`                                                                                                    | `torch.cos(torch.tensor(0.))` → `tensor(1.)`                                               |
|                          | `torch.pow`                                                                                                    | `torch.pow(torch.tensor(2.), 3)` → `tensor(8.)`                                            |
|                          | `torch.round`                                                                                                  | `torch.round(torch.tensor(3.7))` → `tensor(4.)`                                            |
| **Statistical Operations** | `torch.mean`                                                                                                  | `torch.mean(torch.tensor([1., 2., 3.]))` → `tensor(2.)`                                    |
|                          | `torch.std`                                                                                                    | `torch.std(torch.tensor([1., 2., 3.]))` → `tensor(~0.816)`                                 |
|                          | `torch.median`                                                                                                 | `torch.median(torch.tensor([1, 2, 3]))` → `tensor(2)`                                      |
|                          | `torch.var`                                                                                                    | `torch.var(torch.tensor([1., 2., 3.]))` → `tensor(~0.666)`                                 |
|                          | `torch.mean` (weighted)                                                                                        | `torch.mean(torch.tensor([1., 2.]) * torch.tensor([1., 2.]))` → manual weighted avg       |
|                          | `torch.quantile`                                                                                               | `torch.quantile(torch.tensor([1., 2., 3., 4.]), 0.5)` → `tensor(2.5)`                      |
|                          | `torch.corrcoef`                                                                                               | `torch.corrcoef(torch.tensor([[1, 2], [2, 4]]))` → correlation matrix                      |
|                          | `torch.cov`                                                                                                    | `torch.cov(torch.tensor([[1, 2], [2, 4]]))` → covariance matrix                            |
|                          | `torch.argmin`                                                                                                 | `torch.argmin(torch.tensor([3, 1, 2]))` → `tensor(1)` (index of min)                       |
|                          | `torch.argmax`                                                                                                 | `torch.argmax(torch.tensor([1, 3, 2]))` → `tensor(1)` (index of max)                       |
| **Polynomial Functions** | `torch.polyval`                                                                                                | `torch.polyval(torch.tensor([1, 2]), torch.tensor(2))` → `tensor(4)` (1x + 2 at x=2)       |
|                          | (No direct `poly1d`)                                                                                           | Use `torch.tensor` with manual polynomial evaluation                                       |
|                          | (No direct `polyfit`)                                                                                          | Use `torch.linalg.lstsq` for polynomial fitting                                            |
|                          | (No direct `polyder`)                                                                                          | Manual derivative via `torch.autograd`                                                     |
|                          | (No direct `polyint`)                                                                                          | Manual integration or symbolic tools                                                       |
|                          | (No direct `polymul`)                                                                                          | Manual multiplication of polynomial coefficients                                           |
|                          | (No direct `polyadd`)                                                                                          | Manual addition of polynomial coefficients                                                 |
|                          | (No direct `polysub`)                                                                                          | Manual subtraction of polynomial coefficients                                              |
| **Random Number Generation** | `torch.rand`                                                                                               | `torch.rand(2)` → `tensor([0.12, 0.89])` (random floats 0-1)                               |
|                          | `torch.randn`                                                                                                  | `torch.randn(2)` → `tensor([0.5, -1.2])` (standard normal)                                 |
|                          | `torch.randint`                                                                                                | `torch.randint(0, 5, (2,))` → `tensor([3, 1])` (random ints 0-4)                           |
|                          | `torch.rand` (uniform)                                                                                         | `torch.rand(2) * (1-0) + 0` → `tensor([0.45, 0.67])` (uniform random)                      |
|                          | `torch.multinomial`                                                                                            | `torch.multinomial(torch.tensor([1., 1., 1.]), 2)` → `tensor([2, 1])` (random pick)        |
|                          | `torch.randperm`                                                                                               | `torch.randperm(3)` → `tensor([2, 0, 1])` (shuffled indices)                               |
|                          | `torch.normal`                                                                                                 | `torch.normal(0, 1, (2,))` → `tensor([0.2, -0.8])` (normal dist.)                          |
|                          | `torch.manual_seed`                                                                                            | `torch.manual_seed(42)` → sets seed for reproducibility                                    |
| **Linear Algebra**       | `torch.linalg.inv`                                                                                             | `torch.linalg.inv(torch.tensor([[1., 2.], [3., 4.]]))` → inverse matrix                    |
|                          | `torch.linalg.det`                                                                                             | `torch.linalg.det(torch.tensor([[1., 2.], [3., 4.]]))` → `tensor(-2.)` (determinant)       |
|                          | `torch.linalg.eig`                                                                                             | `torch.linalg.eig(torch.tensor([[1., 2.], [3., 4.]]))` → eigenvalues, eigenvectors         |
|                          | `torch.linalg.norm`                                                                                            | `torch.linalg.norm(torch.tensor([3., 4.]))` → `tensor(5.)` (L2 norm)                       |
|                          | `torch.linalg.svd`                                                                                             | `torch.linalg.svd(torch.tensor([[1., 2.], [3., 4.]]))` → U, S, Vh tensors                  |
|                          | `torch.linalg.solve`                                                                                           | `torch.linalg.solve(torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([5., 6.]))` → `tensor([1., -2.])` |
|                          | `torch.linalg.matrix_rank`                                                                                     | `torch.linalg.matrix_rank(torch.tensor([[1., 2.], [2., 4.]]))` → `tensor(1)` (rank)        |
|                          | `torch.linalg.eigh`                                                                                            | `torch.linalg.eigh(torch.tensor([[1., 0.], [0., 1.]]))` → eigenvalues, eigenvectors        |

# Examples

```python
import torch

# 1. Creating a Tensor: Converts a list to a PyTorch tensor
tensor = torch.tensor([1, 2, 3, 4])
print("1:", tensor)  # Output: tensor([1, 2, 3, 4])

# 2. Generating Zeros: Creates a tensor of zeros
zeros = torch.zeros(5)
print("2:", zeros)  # Output: tensor([0., 0., 0., 0., 0.])

# 3. Evenly Spaced Numbers: Generates numbers with a step
range_tensor = torch.arange(0, 10, 2)
print("3:", range_tensor)  # Output: tensor([0, 2, 4, 6, 8])

# 4. Reshaping a Tensor: Changes tensor shape to 2x3
tensor = torch.tensor([1, 2, 3, 4, 5, 6])
reshaped = torch.reshape(tensor, (2, 3))
print("4:", reshaped)  # Output: tensor([[1, 2, 3], [4, 5, 6]])

# 5. Matrix Multiplication: Computes dot product
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
dot_product = torch.matmul(a, b)
print("5:", dot_product)  # Output: tensor(11)

# 6. Finding the Mean: Calculates average (requires float tensor)
data = torch.tensor([1., 2., 3., 4., 5.])
mean_val = torch.mean(data)
print("6:", mean_val)  # Output: tensor(3.)

# 7. Element-Wise Square Root: Applies sqrt to each element
nums = torch.tensor([4., 9., 16.])
sqrt_nums = torch.sqrt(nums)
print("7:", sqrt_nums)  # Output: tensor([2., 3., 4.])

# 8. Stacking Tensors Vertically: Stacks rows
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
stacked = torch.vstack((a, b))
print("8:", stacked)  # Output: tensor([[1, 2], [3, 4]])

# 9. Random Numbers: Generates random floats (0-1)
torch.manual_seed(42)  # For reproducibility
rand_nums = torch.rand(3)
print("9:", rand_nums)  # Output: tensor([0.1915, 0.6221, 0.4377])

# 10. Maximum Value: Finds the largest value
tensor = torch.tensor([5, 2, 8, 1, 9])
max_val = torch.max(tensor)
print("10:", max_val)  # Output: tensor(9)
```
