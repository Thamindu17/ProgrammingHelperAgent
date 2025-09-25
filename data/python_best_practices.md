# Python Programming Best Practices

This document contains Python programming best practices and coding guidelines that the Programming Helper Agent can reference when providing coding advice.

## Code Style and Formatting

### PEP 8 Guidelines
- Use 4 spaces for indentation (not tabs)
- Limit lines to 79 characters for code, 72 for comments
- Use lowercase with underscores for function and variable names
- Use CamelCase for class names
- Use UPPERCASE for constants

### Example:
```python
# Good
def calculate_total_price(item_price, tax_rate):
    """Calculate total price including tax."""
    return item_price * (1 + tax_rate)

# Bad
def calculateTotalPrice(itemPrice,taxRate):
    return itemPrice*(1+taxRate)
```

## Error Handling

### Use Specific Exception Types
```python
# Good
try:
    value = int(user_input)
except ValueError:
    print("Please enter a valid number")

# Bad
try:
    value = int(user_input)
except:
    print("Something went wrong")
```

### Handle Exceptions Appropriately
- Don't suppress exceptions without good reason
- Log exceptions for debugging
- Use finally blocks for cleanup

## Function Design

### Single Responsibility Principle
Each function should do one thing well:

```python
# Good
def read_file(filename):
    """Read and return file contents."""
    with open(filename, 'r') as f:
        return f.read()

def process_data(data):
    """Process the data and return results."""
    # Processing logic here
    return processed_data

# Bad
def read_and_process_file(filename):
    """Read file and process data."""
    with open(filename, 'r') as f:
        data = f.read()
    # Processing logic here
    return processed_data
```

### Use Type Hints
```python
from typing import List, Dict, Optional

def find_user(users: List[Dict[str, str]], name: str) -> Optional[Dict[str, str]]:
    """Find a user by name."""
    for user in users:
        if user.get('name') == name:
            return user
    return None
```

## Data Structures

### Choose the Right Data Structure
- **Lists**: Ordered, mutable sequences
- **Tuples**: Ordered, immutable sequences
- **Sets**: Unordered, unique elements
- **Dictionaries**: Key-value mappings

### List Comprehensions
```python
# Good
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Less efficient
squares = []
for x in range(10):
    squares.append(x**2)
```

## Object-Oriented Programming

### Class Design Principles
```python
class BankAccount:
    """A simple bank account class."""
    
    def __init__(self, account_number: str, initial_balance: float = 0.0):
        self.account_number = account_number
        self._balance = initial_balance  # Protected attribute
    
    @property
    def balance(self) -> float:
        """Get the current balance."""
        return self._balance
    
    def deposit(self, amount: float) -> None:
        """Deposit money to the account."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
    
    def withdraw(self, amount: float) -> bool:
        """Withdraw money from the account."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self._balance:
            return False  # Insufficient funds
        self._balance -= amount
        return True
```

## Performance Tips

### Use Built-in Functions
```python
# Good
total = sum(numbers)
maximum = max(numbers)

# Less efficient
total = 0
for num in numbers:
    total += num
```

### Generator Expressions for Large Data
```python
# Memory efficient
total = sum(x**2 for x in range(1000000))

# Memory intensive
total = sum([x**2 for x in range(1000000)])
```

## Testing

### Write Unit Tests
```python
import unittest

class TestBankAccount(unittest.TestCase):
    def setUp(self):
        self.account = BankAccount("12345", 100.0)
    
    def test_deposit(self):
        self.account.deposit(50.0)
        self.assertEqual(self.account.balance, 150.0)
    
    def test_withdraw_success(self):
        result = self.account.withdraw(30.0)
        self.assertTrue(result)
        self.assertEqual(self.account.balance, 70.0)
    
    def test_withdraw_insufficient_funds(self):
        result = self.account.withdraw(200.0)
        self.assertFalse(result)
        self.assertEqual(self.account.balance, 100.0)
```

## Documentation

### Write Clear Docstrings
```python
def binary_search(arr: List[int], target: int) -> int:
    """
    Perform binary search on a sorted array.
    
    Args:
        arr: A sorted list of integers
        target: The value to search for
    
    Returns:
        The index of the target if found, -1 otherwise
    
    Raises:
        TypeError: If arr is not a list
    
    Example:
        >>> binary_search([1, 3, 5, 7, 9], 5)
        2
    """
    # Implementation here
    pass
```

## Security Considerations

### Input Validation
```python
def process_user_input(user_data: str) -> str:
    """Process and sanitize user input."""
    if not isinstance(user_data, str):
        raise TypeError("Input must be a string")
    
    # Remove potential harmful characters
    sanitized = user_data.strip()
    
    # Validate length
    if len(sanitized) > 1000:
        raise ValueError("Input too long")
    
    return sanitized
```

### Don't Store Secrets in Code
```python
# Good
import os
api_key = os.getenv('API_KEY')

# Bad
api_key = "secret_key_123"  # Never do this!
```

## Common Python Pitfalls

### Mutable Default Arguments
```python
# Bad
def add_item(item, items=[]):
    items.append(item)
    return items

# Good
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### Late Binding Closures
```python
# Bad
functions = []
for i in range(3):
    functions.append(lambda: i)  # All will return 2

# Good
functions = []
for i in range(3):
    functions.append(lambda x=i: x)  # Each returns correct value
```

## Code Organization

### Module Structure
```
myproject/
├── __init__.py
├── main.py
├── config.py
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   └── validators.py
└── tests/
    ├── __init__.py
    ├── test_main.py
    └── test_utils.py
```

### Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import requests
import numpy as np

# Local imports
from .utils import helper_function
from .config import settings
```

## Conclusion

Following these best practices will make your Python code more:
- Readable and maintainable
- Efficient and performant
- Secure and robust
- Testable and debuggable

Remember: "Code is read much more often than it is written." - Guido van Rossum