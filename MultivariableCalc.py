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

def symbolic_vector_magnitude(vector, var_name='x'):
    """
    Calculates the magnitude of a vector whose components are equations in a variable.

    Args:
        vector (list of str): Components as strings, e.g. ["x+1", "2*x"]
        var_name (str): The variable used in the equations, default is 'x'.

    Returns:
        sympy expression: Symbolic magnitude (can be further evaluated).
    """
    var = sympy.symbols(var_name)
    comp_exprs = [sympy.sympify(comp) for comp in vector]
    magnitude = sympy.sqrt(sum(comp**2 for comp in comp_exprs))
    return magnitude

def vector_derivative(vector_func, var):
    x = sympy.symbols(var)
    derivatives = []
    for func_str in vector_func:
        func = sympy.sympify(func_str)       # Convert string to sympy expression
        deriv = sympy.diff(func, x)          # Differentiate with respect to var
        derivatives.append(str(deriv)) # Convert back to string
    return derivatives

def unit_tangent_vector(vector_valued_function):
    # Compute the derivative of the vector valued function
    derivative = vector_derivative(vector_valued_function, "t")
    # Compute the magnitude of the derivative
    magnitude = symbolic_vector_magnitude(derivative)
    # Compute the unit tangent vector
    unit_tangent = [f"({component})/({magnitude})" for component in derivative]
    return unit_tangent

def mv_evaluate_expression_at_value(expr_str, value, var='x'):
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
    result = []
    for i in range(len(expr_str)):
        expr = sympy.sympify(expr_str[i])
        result.append(expr.subs(x, value))
    return result


def cross_product_str_to_str(vec1: list[str], vec2: list[str]) -> list[str]:
    if len(vec1) != 3 or len(vec2) != 3:
        raise ValueError("Both input vectors must have exactly 3 components.")

    # Convert input strings to sympy expressions
    v1 = [sympy.sympify(e) for e in vec1]
    v2 = [sympy.sympify(e) for e in vec2]

    # Cross product
    cp_x = v1[1]*v2[2] - v1[2]*v2[1]
    cp_y = v1[2]*v2[0] - v1[0]*v2[2]
    cp_z = v1[0]*v2[1] - v1[1]*v2[0]

    # Convert result back to strings
    return [str(cp_x), str(cp_y), str(cp_z)]


def matrix_operation(matrix1, matrix2, operation):
    if not (len(matrix1) == len(matrix2) and all(len(row1) == len(row2) for row1, row2 in zip(matrix1, matrix2))):
        raise ValueError("Matrices must have the same dimensions.")

    if operation == "addition":
        return [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(matrix1, matrix2)]
    elif operation == "subtraction":
        return [[a - b for a, b in zip(row1, row2)] for row1, row2 in zip(matrix1, matrix2)]
    else:
        raise ValueError("Operation must be 'addition' or 'subtraction'.")
    

def matrix_product(matrix1, matrix2):
    n = len(matrix1)
    m = len(matrix1[0])
    if not all(len(row) == m for row in matrix1):
        raise ValueError("All rows in matrix1 must have the same length.")
    
    if len(matrix2) == 0 or len(matrix2[0]) == 0:
        raise ValueError("matrix2 cannot be empty.")
    
    p = len(matrix2[0])
    if not all(len(row) == p for row in matrix2):
        raise ValueError("All rows in matrix2 must have the same length.")

    if m != len(matrix2):
        raise ValueError("Number of columns in matrix1 must equal number of rows in matrix2.")

    # Matrix multiplication
    result = []
    for i in range(n):
        result_row = []
        for j in range(p):
            total = 0
            for k in range(m):
                total += matrix1[i][k] * matrix2[k][j]
            result_row.append(total)
        result.append(result_row)
    return result


def transpose_matrix(matrix):
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise ValueError("Input must be a non-empty 2D array (list of lists).")
    if not all(len(row) == len(matrix[0]) for row in matrix):
        raise ValueError("All rows in the matrix must have the same length.")

    # Use zip to transpose the matrix
    return [list(row) for row in zip(*matrix)]


def determinant_2x2(matrix):
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or not all(isinstance(row, list) and len(row) == 2 for row in matrix)
    ):
        raise ValueError("Input must be a 2x2 matrix as a 2D array.")

    # For matrix [[a, b], [c, d]], determinant is ad - bc
    a, b = matrix[0]
    c, d = matrix[1]
    return a * d - b * c


def determinant_3x3(matrix):
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or not all(isinstance(row, list) and len(row) == 3 for row in matrix)
    ):
        raise ValueError("Input must be a 3x3 matrix as a 2D array.")

    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    # Determinant formula for 3x3 matrix:
    # |A| = a(ei − fh) − b(di − fg) + c(dh − eg)
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def inverse_2x2(matrix):
    det = determinant_2x2(matrix)
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")

    a, b = matrix[0]
    c, d = matrix[1]
    # Inverse formula: (1/det) * [[d, -b], [-c, a]]
    return [
        [d / det, -b / det],
        [-c / det, a / det]
    ]


def inverse_3x3(matrix):
    det = determinant_3x3(matrix)
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")

    # Calculate matrix of cofactors
    cofactors = []
    for i in range(3):
        cofactor_row = []
        for j in range(3):
            # Create minor by removing i-th row and j-th column
            minor = [
                [matrix[x][y] for y in range(3) if y != j]
                for x in range(3) if x != i
            ]
            # Determinant of minor (2x2)
            minor_det = minor[0][0]*minor[1][1] - minor[0][1]*minor[1][0]
            # Apply sign
            sign = (-1) ** (i + j)
            cofactor_row.append(sign * minor_det)
        cofactors.append(cofactor_row)

    # Transpose cofactor matrix to get adjugate
    adjugate = [list(row) for row in zip(*cofactors)]

    # Divide adjugate by determinant
    inverse = [[adjugate[i][j] / det for j in range(3)] for i in range(3)]
    return inverse