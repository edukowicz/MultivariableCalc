import math
import sympy

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


def derivative_as_string(expr_str, var='x'):
    """
    Takes a string representing a mathematical expression and returns
    a string representing its derivative with respect to the specified variable.

    Args:
        expr_str (str): The mathematical expression, e.g., "sin(x) + x**2".
        var (str): The variable to differentiate with respect to. Default is 'x'.

    Returns:
        str: The string of the derivative.
    """
    x = sympy.symbols(var)
    expr = sympy.sympify(expr_str)
    deriv = sympy.diff(expr, x)
    return str(deriv)


def antiderivative_as_string(expr_str, var='x'):
    """
    Takes a string representing a mathematical expression and returns
    a string representing its antiderivative (indefinite integral)
    with respect to the specified variable.

    Args:
        expr_str (str): The mathematical expression, e.g., "cos(x) + x".
        var (str): The variable to integrate with respect to. Default is 'x'.

    Returns:
        str: The string of the antiderivative (without the constant of integration).
    """
    x = sympy.symbols(var)
    expr = sympy.sympify(expr_str)
    anti = sympy.integrate(expr, x)
    return str(anti)


def definite_integral_as_string(expr_str, lower, upper, var='x'):
    """
    Takes a string representing a mathematical expression and two limits of integration,
    returns a string representing the definite integral of the expression
    from lower to upper with respect to the specified variable.

    Args:
        expr_str (str): The mathematical expression, e.g., "x**2 + 1".
        lower (int or float): The lower limit of integration.
        upper (int or float): The upper limit of integration.
        var (str): The variable to integrate with respect to. Default is 'x'.

    Returns:
        str: The string of the definite integral result.
    """
    x = sympy.symbols(var)
    expr = sympy.sympify(expr_str)
    definite_integral = sympy.integrate(expr, (x, lower, upper))
    return str(definite_integral)


def evaluate_expression_at_value(expr_str, value, var='x'):
    """
    Takes a string representing a mathematical expression and a value,
    returns the result of evaluating the expression at that value.

    Args:
        expr_str (str): The mathematical expression, e.g., "x**2 + 1".
        value (float or int): The value at which to evaluate the expression.
        var (str): The variable in the expression. Default is 'x'.

    Returns:
        float: The result of the evaluated expression at the given value.
    """
    x = sympy.symbols(var)
    expr = sympy.sympify(expr_str)
    result = expr.subs(x, value)
    return float(result)

 

def mv_function_integral(expr_list, var_list, int_var): 
    """ 
    Computes the symbolic integral of a function with respect to a variable. 
    Args: 
        expr_list (list of str): The function as a string, e.g., ["x**2 + y*x","sin(x)*y"]. 
        var_list (list of str): List of variable names, e.g., ["x", "y"]. 
        int_var (str): The variable to integrate with respect to, e.g., "x". 
    Returns: 
        sympy expression: The symbolic integral. 
    example: 
        integral = function_integral(["x**2 + y*x","sin(x)*y"], ["x", "y"], "x") 
        for integ in integral 
            print(integ) # Output: x**3/3 + y*x**2/2, -y*cos(x) 
    """ 
    variables = sympy.symbols(var_list) 
    integrals = [] 
    for expr_str in expr_list: 
        expr = sympy.sympify(expr_str) 
        integral = sympy.integrate(expr, sympy.Symbol(int_var)) 
        integrals.append(integral) 
    return integrals 



def mv_function_definite_integral(expr_list, var_list, int_var, lower_limit, upper_limit): 
    """ 
    Computes the definite integral of a function with respect to a variable over given limits. 
    Args: 
        expr_list (list of str): The function as a string, e.g., ["x**2 + y*x","sin(x)*y"]. 
        var_list (list of str): List of variable names, e.g., ["x", "y"]. 
        int_var (str): The variable to integrate with respect to, e.g., "x". 
        lower_limit (float): The lower limit of integration. 
        upper_limit (float): The upper limit of integration. 
    Returns: 
        float: The evaluated definite integral. 
    example: 
        result = function_definite_integral(["x**2 + y*x","sin(x)*y"], ["x", "y"], "x", 0, 1) 
        print(result) # Output: 1/3 + y/2, -y*cos(1) - y 
    """ 
    variables = sympy.symbols(var_list) 
    definite_integrals = [] 
    for expr_str in expr_list: 
        expr = sympy.sympify(expr_str) 
        definite_integral = sympy.integrate(expr, (sympy.Symbol(int_var), lower_limit, upper_limit)) 
        definite_integrals.append(definite_integral.evalf()) 
    return definite_integrals 


def dot_product(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be the same length")
    return sum(a * b for a, b in zip(vec1, vec2))


def cross_product(vec1, vec2):
    if len(vec1) != 3 or len(vec2) != 3:
        raise ValueError("Both vectors must be of length 3.")
    x1, y1, z1 = vec1
    x2, y2, z2 = vec2
    return [
        y1 * z2 - z1 * y2,
        z1 * x2 - x1 * z2,
        x1 * y2 - y1 * x2
    ]