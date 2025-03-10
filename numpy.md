| **Category**             | **Functions**          | **Example**                          |
|--------------------------|------------------------|--------------------------------------|
| **Array Creation**       |                        |                                      |
|                          | `np.array`            | `np.array([1, 2, 3])` → `[1 2 3]`   |
|                          | `np.zeros`            | `np.zeros(3)` → `[0. 0. 0.]`        |
| **Array Manipulation**   |                        |                                      |
|                          | `np.reshape`          | `np.reshape([1, 2, 3, 4], (2, 2))` → `[[1 2] [3 4]]` |
|                          | `np.split`            | `np.split([1, 2, 3, 4], 2)` → `[array([1, 2]), array([3, 4])]` |
| **Mathematical Operations** |                     |                                      |
|                          | `np.dot`              | `np.dot([1, 2], [3, 4])` → `11`     |
|                          | `np.exp`              | `np.exp(1)` → `2.718…`              |
| **Statistical Operations** |                      |                                      |
|                          | `np.mean`             | `np.mean([1, 2, 3])` → `2.0`        |
|                          | `np.std`              | `np.std([1, 2, 3])` → `~0.816`      |
| **Polynomial Functions** |                        |                                      |
|                          | `np.poly1d`           | `p = np.poly1d([1, 2]); p(2)` → `4` |
|                          | `np.polyfit`          | `np.polyfit([1, 2], [2, 4], 1)` → `[2. 0.]` |
| **Random Number Generation** |                    |                                      |
|                          | `np.random.rand`      | `np.random.rand(2)` → `[0.12 0.89]` |
|                          | `np.random.randn`     | `np.random.randn(2)` → `[0.5 -1.2]` |
| **Linear Algebra**       |                        |                                      |
|                          | `np.linalg.inv`       | `np.linalg.inv([[1, 2], [3, 4]])` → inverse matrix |
|                          | `np.linalg.det`       | `np.linalg.det([[1, 2], [3, 4]])` → `-2.0` |
