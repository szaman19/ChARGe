from typing import Callable

def verifier(func: Callable) -> Callable:
    """Decorator to mark a method as a verifier.
    
    Args:
        func (Callable): The function to mark as a verifier.
    
    Returns:
        Callable: The function marked as a verifier.
    """
    func.__verifier_tag = True
    return func


def is_verifier(func: Callable) -> bool:
    """Check if a method is marked as a verifier.
    
    Args:
        func (Callable): The function to check.
    
    Returns:
        bool: True if the function is marked as a verifier, False otherwise.
    """
    return hasattr(func, "__verifier_tag")


def hypothesis(func: Callable) -> Callable:
    """Decorator to mark a method as a hypothesis.
    
    Args:
        func (Callable): The function to mark as a hypothesis.
    
    Returns:
        Callable: The function marked as a hypothesis.

    
    Side Effects:
        Marks the function as a hypothesis. 
        `func` is modified in place to include the `__hypothesis_tag` attribute.
    """
    func.__hypothesis_tag = True
    return func


def is_hypothesis(func: Callable) -> bool:
    """Check if a method is marked as a hypothesis.
    
    Args:
        func (Callable): The function to check.
    
    Returns:
        bool: True if the function is marked as a hypothesis, False otherwise.
    """
    return hasattr(func, "__hypothesis_tag")
