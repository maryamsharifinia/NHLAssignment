# NHLAssignment

# Debugging Exercises — Answer Explanations

---

## Exercise 1 — `id_to_fruit`

### The Bug

The original function used a `set` to store fruits and manually tracked an index with a loop:

```python
def id_to_fruit(fruit_id: int, fruits: Set[str]) -> str:
    idx = 0
    for fruit in fruits:
        if fruit_id == idx:
            return fruit
        idx += 1
```

**Problem:** `set` in Python is **unordered**. Every time you iterate over a set, the order of elements may differ. So `idx == 1` does not reliably point to `'orange'` — or any specific fruit.

### Fix

Change the input type from `Set[str]` to `list[str]` and return directly by index:

```python
def id_to_fruit(fruit_id: int, fruits: list[str]) -> str:
    return fruits[fruit_id]
```

**Why it works:** A `list` is ordered. Index `1` always refers to the second element, exactly as expected.

---

## Exercise 2 — `swap` (flipping x and y coordinates)

### The Bugs

The original function had two bugs:

```python
coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3] = \
    coords[:, 1], coords[:, 1], coords[:, 3], coords[:, 2]
#              ^^^
#        Typo! Should be coords[:, 0]
```

**Bug 1 (obvious):** Column 0 should receive `coords[:, 0]` (x1), but `coords[:, 1]` is written twice instead.

**Bug 2 (subtle):** Even after fixing the typo, the result is still wrong. NumPy does not guarantee that the right-hand side values are fully read before any assignment happens on the left-hand side when operating in-place on the same array. Some values get overwritten before they are read, corrupting the result.

### Fix

Take a **copy** of the array first, then perform the swap on the copy:

```python
def swap(coords):
    swapped = coords.copy()
    swapped[:, [0, 1, 2, 3]] = swapped[:, [1, 0, 3, 2]]
    return swapped
```

**Why it works:** `.copy()` preserves the original values. The vectorized column reassignment then reads all values from the untouched copy, so there is no read-before-write conflict.

---

## Exercise 3 — `plot_data` (Precision-Recall curve)

### The Bug

The original function stored CSV rows as **strings** and passed them directly to NumPy and Matplotlib:

```python
for row in csv_reader:
    results.append(row)   # each row is a list of strings
results = np.stack(results)
# results is a string array, not float!
```

**Problem:** `np.stack` on string data creates a dtype `str` array. When Matplotlib plots this, it treats the values as categories (lexicographic order) rather than real numbers, producing an incorrect plot.

### Fix

Explicitly convert each value to `float` before plotting:

```python
def plot_data(csv_file_path: str):
    results = []
    with open(csv_file_path) as result_csv:
        csv_reader = csv.reader(result_csv, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            if len(row) != 0:
                results.append(row)

    # convert to float
    results_f = [[float(element) for element in sublist] for sublist in results]
    results_f = np.array(results_f)

    plt.plot(results_f[:, 1], results_f[:, 0])
    ...
```

**Why it works:** Converting to `float` gives NumPy a proper numeric array, so Matplotlib plots the axes correctly. The `len(row) != 0` check also guards against any empty trailing rows in the CSV.

---

## Exercise 4 — `train_gan` (GAN training loop)

### Bug 1 — Structural Bug

The original code used a **fixed** `batch_size` constant to create label tensors:

```python
real_samples_labels = torch.ones((batch_size, 1))
```

The last batch of each epoch often contains fewer samples than `batch_size` (because the dataset size is rarely an exact multiple of `batch_size`). This causes a shape mismatch between the input and the labels:

```
ValueError: Using a target size (torch.Size([128, 1])) that is different
to the input size (torch.Size([96, 1]))
```

### Bug 2 — Cosmetic Bug

The condition for displaying generated images was wrong:

```python
if n == batch_size - 1:   # incorrect
```

This checks whether the batch index equals `batch_size - 1` (e.g. 63), which has nothing to do with the last batch of the epoch. It may fire too early or not at all depending on the dataset size.

### Fix

**Structural fix:** Use `real_samples.size(0)` instead of the fixed `batch_size` so the label tensor always matches the actual batch size:

```python
real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device=device)        # dynamic size
generated_samples_labels = torch.zeros((generated_samples.size(0), 1)).to(device=device)
generator_labels = torch.ones((generated_samples.size(0), 1)).to(device=device)
```

**Cosmetic fix:** Use `len(train_loader) - 1` to correctly identify the last batch of each epoch:

```python
if n == len(train_loader) - 1:   # last batch of the epoch
```

**Why it works:**
- `real_samples.size(0)` returns the actual number of samples in the current batch, which may be smaller than `batch_size` for the final batch.
- `len(train_loader)` is the total number of batches, so `len(train_loader) - 1` is the correct index of the last batch.
