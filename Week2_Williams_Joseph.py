"""
Array vs. List Performance Benchmark

This script compares Python lists and NumPy arrays for:
1. Insertion
2. Lookup
3. Deletion

Results are displayed in milliseconds and saved to timing_results.csv.
"""

import csv
import gc
import statistics
import time
import numpy as np


# Input sizes to benchmark. Adjust if your computer is slow.
SIZES = [1_000, 10_000, 100_000, 1_000_000]

# Number of lookup operations performed per lookup test.
LOOKUP_COUNT = 10_000


def median_time(operation, make_data, repeats=5):
    """
    Measures the median time required to perform an operation.

    The data structure is copied before each timed trial so that each trial
    begins with the same original data.
    """
    times = []

    gc_was_enabled = gc.isenabled()
    gc.disable()

    try:
        for _ in range(repeats):
            data = make_data()

            start_time = time.perf_counter()
            result = operation(data)
            end_time = time.perf_counter()

            # Keep the result referenced so the operation is completed.
            _ = result
            times.append(end_time - start_time)
    finally:
        if gc_was_enabled:
            gc.enable()

    return statistics.median(times)


# ---------- List Operations ----------

def list_insert(data, position):
    """Insert one value into the middle of a Python list."""
    data.insert(position, -1)
    return data


def list_lookup(data, indices):
    """Look up multiple positions in a Python list."""
    checksum = 0

    for index in indices:
        checksum += data[index]

    return checksum


def list_delete(data, position):
    """Delete one value from the middle of a Python list."""
    del data[position]
    return data


# ---------- NumPy Array Operations ----------

def array_insert(data, position):
    """
    Insert one value into a NumPy array.

    np.insert returns a new array because NumPy arrays have fixed sizes.
    """
    return np.insert(data, position, -1)


def array_lookup(data, indices):
    """Look up multiple positions in a NumPy array."""
    checksum = 0

    for index in indices:
        checksum += int(data[index])

    return checksum


def array_delete(data, position):
    """
    Delete one value from a NumPy array.

    np.delete returns a new array because NumPy arrays have fixed sizes.
    """
    return np.delete(data, position)


def print_results(results):
    """Print benchmark results in a readable table."""
    print("\nARRAY VS. LIST PERFORMANCE RESULTS")
    print("-" * 76)
    print(f"{'Size':>12} {'Structure':>12} {'Operation':>12} {'Time (ms)':>15}")
    print("-" * 76)

    for row in results:
        print(
            f"{row['size']:>12,} "
            f"{row['structure']:>12} "
            f"{row['operation']:>12} "
            f"{row['time_ms']:>15.4f}"
        )

    print("-" * 76)


def save_results_to_csv(results, filename="timing_results.csv"):
    """Save benchmark results to a CSV file."""
    with open(filename, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["size", "structure", "operation", "time_ms"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {filename}")


def main():
    results = []

    for size in SIZES:
        print(f"Benchmarking size: {size:,}")

        # Use fewer repeats for very large structures to reduce runtime.
        repeats = 5 if size <= 100_000 else 3

        # Create original structures.
        base_list = list(range(size))
        base_array = np.arange(size, dtype=np.int64)

        # Use the middle position because insertion/deletion there requires
        # shifting approximately half of the elements.
        middle_position = size // 2

        # Create repeatable lookup indices distributed across the structure.
        lookup_count = min(LOOKUP_COUNT, size)
        lookup_indices = [(i * 7919) % size for i in range(lookup_count)]

        # ----- List benchmarks -----
        list_insert_time = median_time(
            lambda data: list_insert(data, middle_position),
            lambda: base_list.copy(),
            repeats
        )

        list_lookup_time = median_time(
            lambda data: list_lookup(data, lookup_indices),
            lambda: base_list.copy(),
            repeats
        )

        list_delete_time = median_time(
            lambda data: list_delete(data, middle_position),
            lambda: base_list.copy(),
            repeats
        )

        # ----- NumPy array benchmarks -----
        array_insert_time = median_time(
            lambda data: array_insert(data, middle_position),
            lambda: base_array.copy(),
            repeats
        )

        array_lookup_time = median_time(
            lambda data: array_lookup(data, lookup_indices),
            lambda: base_array.copy(),
            repeats
        )

        array_delete_time = median_time(
            lambda data: array_delete(data, middle_position),
            lambda: base_array.copy(),
            repeats
        )

        # Store results in milliseconds.
        results.extend([
            {
                "size": size,
                "structure": "Python List",
                "operation": "Insertion",
                "time_ms": list_insert_time * 1000
            },
            {
                "size": size,
                "structure": "Python List",
                "operation": "Lookup",
                "time_ms": list_lookup_time * 1000
            },
            {
                "size": size,
                "structure": "Python List",
                "operation": "Deletion",
                "time_ms": list_delete_time * 1000
            },
            {
                "size": size,
                "structure": "NumPy Array",
                "operation": "Insertion",
                "time_ms": array_insert_time * 1000
            },
            {
                "size": size,
                "structure": "NumPy Array",
                "operation": "Lookup",
                "time_ms": array_lookup_time * 1000
            },
            {
                "size": size,
                "structure": "NumPy Array",
                "operation": "Deletion",
                "time_ms": array_delete_time * 1000
            }
        ])

    print_results(results)
    save_results_to_csv(results)

'''
Benchmarking size: 1,000
Benchmarking size: 10,000
Benchmarking size: 100,000
Benchmarking size: 1,000,000

ARRAY VS. LIST PERFORMANCE RESULTS
----------------------------------------------------------------------------
        Size    Structure    Operation       Time (ms)
----------------------------------------------------------------------------
       1,000  Python List    Insertion          0.0024
       1,000  Python List       Lookup          0.0533
       1,000  Python List     Deletion          0.0019
       1,000  NumPy Array    Insertion          0.0305
       1,000  NumPy Array       Lookup          0.2101
       1,000  NumPy Array     Deletion          0.0144
      10,000  Python List    Insertion          0.0183
      10,000  Python List       Lookup          0.5599
      10,000  Python List     Deletion          0.0363
      10,000  NumPy Array    Insertion          0.0422
      10,000  NumPy Array       Lookup          1.6986
      10,000  NumPy Array     Deletion          0.0133
     100,000  Python List    Insertion          0.2282
     100,000  Python List       Lookup          0.8534
     100,000  Python List     Deletion          0.3885
     100,000  NumPy Array    Insertion          0.0974
     100,000  NumPy Array       Lookup          1.7528
     100,000  NumPy Array     Deletion          0.0935
   1,000,000  Python List    Insertion          4.1749
   1,000,000  Python List       Lookup          2.7162
   1,000,000  Python List     Deletion          5.9138
   1,000,000  NumPy Array    Insertion          1.1387
   1,000,000  NumPy Array       Lookup          2.1051
   1,000,000  NumPy Array     Deletion          1.2427
----------------------------------------------------------------------------

Results saved to: timing_results.csv
'''

if __name__ == "__main__":
    main()
