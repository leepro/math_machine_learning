# NumPy Cheatsheet

| **Category**             | **Functions**                                                                                                   | **Example**                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **Array Creation**       | `np.array`                                                                                                     | `np.array([1, 2, 3])` → `[1 2 3]`                                                           |
|                          | `np.zeros`                                                                                                     | `np.zeros(3)` → `[0. 0. 0.]`                                                                |
|                          | `np.ones`                                                                                                      | `np.ones(2)` → `[1. 1.]`                                                                    |
|                          | `np.arange`                                                                                                    | `np.arange(0, 5, 2)` → `[0 2 4]`                                                           |
|                          | `np.linspace`                                                                                                  | `np.linspace(0, 1, 3)` → `[0. 0.5 1.]`                                                     |
|                          | `np.eye`                                                                                                       | `np.eye(2)` → `[[1. 0.] [0. 1.]]`                                                          |
|                          | `np.full`                                                                                                      | `np.full(3, 7)` → `[7 7 7]`                                                                |
|                          | `np.empty`                                                                                                     | `np.empty(2)` → `[random values]` (uninitialized)                                           |
|                          | `np.diag`                                                                                                      | `np.diag([1, 2])` → `[[1 0] [0 2]]`                                                        |
| **Array Manipulation**   | `np.reshape`                                                                                                   | `np.reshape([1, 2, 3, 4], (2, 2))` → `[[1 2] [3 4]]`                                       |
|                          | `np.split`                                                                                                     | `np.split([1, 2, 3, 4], 2)` → `[array([1, 2]), array([3, 4])]`                             |
|                          | `np.vstack`                                                                                                    | `np.vstack(([1, 2], [3, 4]))` → `[[1 2] [3 4]]`                                            |
|                          | `np.hstack`                                                                                                    | `np.hstack(([1, 2], [3, 4]))` → `[1 2 3 4]`                                                |
|                          | `np.tile`                                                                                                      | `np.tile([1, 2], 2)` → `[1 2 1 2]`                                                         |
|                          | `np.delete`                                                                                                    | `np.delete([1, 2, 3], 1)` → `[1 3]`                                                        |
|                          | `np.append`                                                                                                    | `np.append([1, 2], 3)` → `[1 2 3]`                                                         |
|                          | `np.insert`                                                                                                    | `np.insert([1, 2], 1, 5)` → `[1 5 2]`                                                      |
|                          | `np.pad`                                                                                                       | `np.pad([1, 2], (1, 1), 'constant')` → `[0 1 2 0]`                                         |
|                          | `np.concatenate`                                                                                               | `np.concatenate(([1, 2], [3, 4]))` → `[1 2 3 4]`                                           |
|                          | `np.flip`                                                                                                      | `np.flip([1, 2, 3])` → `[3 2 1]`                                                           |
|                          | `np.transpose`                                                                                                 | `np.transpose([[1, 2], [3, 4]])` → `[[1 3] [2 4]]`                                         |
|                          | `np.squeeze`                                                                                                   | `np.squeeze([[1], [2]])` → `[1 2]`                                                         |
| **Mathematical Operations** | `np.dot`                                                                                                    | `np.dot([1, 2], [3, 4])` → `11` (1×3 + 2×4)                                                |
|                          | `np.exp`                                                                                                       | `np.exp(1)` → `2.718…`                                                                      |
|                          | `np.sqrt`                                                                                                      | `np.sqrt(4)` → `2.0`                                                                        |
|                          | `np.max`                                                                                                       | `np.max([1, 3, 2])` → `3`                                                                  |
|                          | `np.min`                                                                                                       | `np.min([1, 3, 2])` → `1`                                                                  |
|                          | `np.sum`                                                                                                       | `np.sum([1, 2, 3])` → `6`                                                                  |
|                          | `np.prod`                                                                                                      | `np.prod([2, 3])` → `6`                                                                    |
|                          | `np.abs`                                                                                                       | `np.abs(-5)` → `5`                                                                          |
|                          | `np.log`                                                                                                       | `np.log(2.718)` → `~1.0`                                                                   |
|                          | `np.sin`                                                                                                       | `np.sin(0)` → `0.0`                                                                         |
|                          | `np.cos`                                                                                                       | `np.cos(0)` → `1.0`                                                                         |
|                          | `np.power`                                                                                                     | `np.power(2, 3)` → `8`                                                                     |
|                          | `np.round`                                                                                                     | `np.round(3.7)` → `4.0`                                                                    |
| **Statistical Operations** | `np.mean`                                                                                                    | `np.mean([1, 2, 3])` → `2.0`                                                               |
|                          | `np.std`                                                                                                       | `np.std([1, 2, 3])` → `~0.816`                                                             |
|                          | `np.median`                                                                                                    | `np.median([1, 2, 3])` → `2.0`                                                             |
|                          | `np.var`                                                                                                       | `np.var([1, 2, 3])` → `~0.666`                                                             |
|                          | `np.average`                                                                                                   | `np.average([1, 2], weights=[1, 2])` → `1.666…`                                            |
|                          | `np.percentile`                                                                                                | `np.percentile([1, 2, 3, 4], 50)` → `2.5`                                                  |
|                          | `np.corrcoef`                                                                                                  | `np.corrcoef([1, 2], [2, 4])` → correlation matrix                                         |
|                          | `np.cov`                                                                                                       | `np.cov([1, 2], [2, 4])` → covariance matrix                                               |
|                          | `np.argmin`                                                                                                    | `np.argmin([3, 1, 2])` → `1` (index of min)                                                |
|                          | `np.argmax`                                                                                                    | `np.argmax([1, 3, 2])` → `1` (index of max)                                                |
| **Polynomial Functions** | `np.poly1d`                                                                                                    | `p = np.poly1d([1, 2]); p(2)` → `4` (1x + 2 at x=2)                                        |
|                          | `np.polyfit`                                                                                                   | `np.polyfit([1, 2], [2, 4], 1)` → `[2. 0.]` (slope, intercept)                             |
|                          | `np.polyval`                                                                                                   | `np.polyval([1, 2], 2)` → `4` (1x + 2 at x=2)                                              |
|                          | `np.polyder`                                                                                                   | `p = np.poly1d([1, 2, 3]); np.polyder(p)` → `[2 2]` (derivative)                           |
|                          | `np.polyint`                                                                                                   | `p = np.poly1d([2, 2]); np.polyint(p)` → `[1 2 0]` (integral)                              |
|                          | `np.polymul`                                                                                                   | `np.polymul([1, 1], [1, 1])` → `[1 2 1]` (x+1 × x+1)                                       |
|                          | `np.polyadd`                                                                                                   | `np.polyadd([1, 1], [1, 1])` → `[2 2]` (x+1 + x+1)                                         |
|                          | `np.polysub`                                                                                                   | `np.polysub([1, 2], [1, 1])` → `[0 1]` (x+2 - (x+1))                                       |
| **Random Number Generation** | `np.random.rand`                                                                                           | `np.random.rand(2)` → `[0.12 0.89]` (random floats 0-1)                                     |
|                          | `np.random.randn`                                                                                              | `np.random.randn(2)` → `[0.5 -1.2]` (standard normal)                                       |
|                          | `np.random.randint`                                                                                            | `np.random.randint(0, 5, 2)` → `[3 1]` (random ints 0-4)                                    |
|                          | `np.random.uniform`                                                                                            | `np.random.uniform(0, 1, 2)` → `[0.45 0.67]` (uniform random)                               |
|                          | `np.random.choice`                                                                                             | `np.random.choice([1, 2, 3], 2)` → `[2 1]` (random pick)                                    |
|                          | `np.random.shuffle`                                                                                            | `a = np.array([1, 2, 3]); np.random.shuffle(a)` → `a = [3 1 2]` (shuffled in place)         |
|                          | `np.random.normal`                                                                                             | `np.random.normal(0, 1, 2)` → `[0.2 -0.8]` (normal dist.)                                   |
|                          | `np.random.seed`                                                                                               | `np.random.seed(42)` → sets seed for reproducibility                                        |
| **Linear Algebra**       | `np.linalg.inv`                                                                                                | `np.linalg.inv([[1, 2], [3, 4]])` → inverse matrix                                         |
|                          | `np.linalg.det`                                                                                                | `np.linalg.det([[1, 2], [3, 4]])` → `-2.0` (determinant)                                   |
|                          | `np.linalg.eig`                                                                                                | `np.linalg.eig([[1, 2], [3, 4]])` → eigenvalues, eigenvectors                              |
|                          | `np.linalg.norm`                                                                                               | `np.linalg.norm([3, 4])` → `5.0` (L2 norm)                                                 |
|                          | `np.linalg.svd`                                                                                                | `np.linalg.svd([[1, 2], [3, 4]])` → U, S, Vh matrices                                       |
|                          | `np.linalg.solve`                                                                                              | `np.linalg.solve([[1, 2], [3, 4]], [5, 6])` → `[1 -2]` (solves Ax = b)                     |
|                          | `np.linalg.matrix_rank`                                                                                        | `np.linalg.matrix_rank([[1, 2], [2, 4]])` → `1` (rank)                                     |
|                          | `np.linalg.eigh`                                                                                               | `np.linalg.eigh([[1, 0], [0, 1]])` → eigenvalues, eigenvectors (Hermitian)                 |

# Examples

```python
import numpy as np

# 1. Creating an Array: Converts a list to a NumPy array
arr = np.array([1, 2, 3, 4])
print("1:", arr)  # Output: [1 2 3 4]

# 2. Generating Zeros: Creates an array of zeros
zeros = np.zeros(5)
print("2:", zeros)  # Output: [0. 0. 0. 0. 0.]

# 3. Evenly Spaced Numbers: Generates numbers with a step
range_arr = np.arange(0, 10, 2)
print("3:", range_arr)  # Output: [0 2 4 6 8]

# 4. Reshaping an Array: Changes array shape to 2x3
arr = np.array([1, 2, 3, 4, 5, 6])
reshaped = np.reshape(arr, (2, 3))
print("4:", reshaped)  # Output: [[1 2 3] [4 5 6]]

# 5. Matrix Multiplication: Computes dot product
a = np.array([1, 2])
b = np.array([3, 4])
dot_product = np.dot(a, b)
print("5:", dot_product)  # Output: 11

# 6. Finding the Mean: Calculates average
data = np.array([1, 2, 3, 4, 5])
mean_val = np.mean(data)
print("6:", mean_val)  # Output: 3.0

# 7. Element-Wise Square Root: Applies sqrt to each element
nums = np.array([4, 9, 16])
sqrt_nums = np.sqrt(nums)
print("7:", sqrt_nums)  # Output: [2. 3. 4.]

# 8. Stacking Arrays Vertically: Stacks rows
a = np.array([1, 2])
b = np.array([3, 4])
stacked = np.vstack((a, b))
print("8:", stacked)  # Output: [[1 2] [3 4]]

# 9. Random Numbers: Generates random floats (0-1)
np.random.seed(42)  # For reproducibility
rand_nums = np.random.rand(3)
print("9:", rand_nums)  # Output: [0.37454012 0.95071431 0.73199394]

# 10. Maximum Value: Finds the largest value
arr = np.array([5, 2, 8, 1, 9])
max_val = np.max(arr)
print("10:", max_val)  # Output: 9
```
