import math

def vector_operation(vec1, vec2, operation):
    """
    Performs addition or subtraction on two input vectors (lists or arrays).
    
    Parameters:
        vec1 (list): First vector.
        vec2 (list): Second vector.
        operation (str): Either "addition" or "subtraction".
        
    Returns:
        list: Resulting vector after the operation, or
        str: Error message if vector lengths do not match or operation is invalid.
    """
    if len(vec1) != len(vec2):
        return "Error: Vector lengths do not match."
    
    if operation == "addition":
        return [a + b for a, b in zip(vec1, vec2)]
    elif operation == "subtraction":
        return [a - b for a, b in zip(vec1, vec2)]
    else:
        return "Error: Invalid operation. Use 'addition' or 'subtraction'."
    

def scalar_multiply(vector, scalar):
    """
    Multiplies each element of the input vector by the scalar value.
    
    Parameters:
        vector (list): The input vector (list of numbers).
        scalar (float or int): The scalar value to multiply.
        
    Returns:
        list: The resulting vector after multiplication.
    """
    return [scalar * x for x in vector]


def unit_vector(vector):
    """
    Returns the unit vector (normalized vector) of the input vector.
    
    Parameters:
        vector (list): The input vector (list of numbers).
        
    Returns:
        list: The unit vector, or
        str: Error message if the vector has zero magnitude.
    """
    magnitude = math.sqrt(sum(x**2 for x in vector))
    if magnitude == 0:
        return "Error: Zero vector cannot be normalized."
    return [x / magnitude for x in vector]


def create_vector(vector, magnitude):
    unit = unit_vector(vector)
    return [magnitude * x for x in unit]