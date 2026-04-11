import os
from typing import List

def calculate_fibonacci(n: int) -> List[int]:
    """
    Calculates the first n Fibonacci numbers.

    Args:
        n: The number of Fibonacci numbers to generate.

    Returns:
        A list of the first n Fibonacci numbers.
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    else:
        fib_sequence = [0, 1]
        while len(fib_sequence) < n:
            next_fib = fib_sequence[-1] + fib_sequence[-2]
            fib_sequence.append(next_fib)
        return fib_sequence

def write_fibonacci_to_file(filepath: str, numbers: List[int]) -> None:
    """
    Writes a list of numbers to a specified file, one number per line.

    Args:
        filepath: The path to the file where numbers will be written.
        numbers: A list of integers to write to the file.
    """
    try:
        with open(filepath, 'w') as f:
            for number in numbers:
                f.write(str(number) + '\n')
        print(f"Successfully wrote Fibonacci numbers to {filepath}")
    except IOError as e:
        print(f"Error writing to file {filepath}: {e}")

def read_fibonacci_from_file(filepath: str) -> List[int]:
    """
    Reads numbers from a specified file, expecting one number per line.

    Args:
        filepath: The path to the file from which numbers will be read.

    Returns:
        A list of integers read from the file.
    """
    numbers: List[int] = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    numbers.append(int(line.strip()))
                except ValueError:
                    print(f"Warning: Could not convert line to integer: '{line.strip()}'")
        print(f"Successfully read Fibonacci numbers from {filepath}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except IOError as e:
        print(f"Error reading from file {filepath}: {e}")
    return numbers

def main():
    """
    Main function to orchestrate Fibonacci calculation, writing to file, and reading from file.
    """
    fib_filepath = "fib.txt"
    num_fibonacci = 10

    # 1. Calculate Fibonacci numbers
    fib_numbers = calculate_fibonacci(num_fibonacci)
    print(f"Generated {num_fibonacci} Fibonacci numbers: {fib_numbers}")

    # 2. Persist them to a text file
    write_fibonacci_to_file(fib_filepath, fib_numbers)

    # 3. Retrieve them from the same file for display
    read_numbers = read_fibonacci_from_file(fib_filepath)
    if read_numbers:
        print(f"Retrieved Fibonacci numbers from file: {read_numbers}")
    else:
        print("No numbers were retrieved from the file.")

    # Optional: Clean up the generated file
    # try:
    #     if os.path.exists(fib_filepath):
    #         os.remove(fib_filepath)
    #         print(f"Cleaned up {fib_filepath}")
    # except OSError as e:
    #     print(f"Error removing file {fib_filepath}: {e}")

if __name__ == "__main__":
    main()
