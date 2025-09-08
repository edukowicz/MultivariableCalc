import math


#funtion "vector_operation" takes in two vectors -- a and b -- and an operation "add" or "subtract"
def vector_operation(vec1, vec2, operation):
 
    #returns an error message if the lengths of the array representing the vectors are not equal
    if len(vec1) != len(vec2):
        return "Error: Vector lengths do not match."
    
    #returns resulting vectors after the operation, or an error message if the operation parameter is not valid
    if operation == "addition":
        return [a + b for a, b in zip(vec1, vec2)]
    elif operation == "subtraction":
        return [a - b for a, b in zip(vec1, vec2)]
    else:
        return "Error: Invalid operation. Use 'addition' or 'subtraction'."
    

#function "scalar_multiply" takes in a vector and a scalar value
def scalar_multiply(vector, scalar):
 
    #multiples each element of the vector by the scalar value
    #returns the resulting vector after multiplication
    return [scalar * x for x in vector]


#function "unit_vector" takes in one vector
def unit_vector(vector):
   
    #set variable "magnitude" as the magnitude of input vector
    #return an error message if the magnitude is zero
    magnitude = math.sqrt(sum(x**2 for x in vector))
    if magnitude == 0:
        return "Error: Zero vector cannot be normalized."
    
    #divides each elements of the input vector by the vector's magnitude
    #returns the unit vector of the input vector
    return [x / magnitude for x in vector]


#function "vector_magnitude" takes in a vector
def vector_magnitude(vector):
   
    #calculates magnitude of input vector
    #returns the magnitude
    return math.sqrt(sum(x**2 for x in vector))