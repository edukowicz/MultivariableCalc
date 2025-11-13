import math
import sympy
import numpy as np
import matplotlib.pyplot as plt

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


def plot_equation(equation_str, x_range=(-10, 10), num_points=1000, ax=None, **kwargs):
    #plots the 2D function y=f(x) specified as a string
    #equation_str (str): The equation as a Python-valid string, e.g., "np.sin(x)".
    #x_range (tuple): Range for x values, default (-10, 10).
    #num_points (int): Number of points to plot, default 1000.
    #ax (matplotlib.axes._axes.Axes or None): If provided, plot on this axes, otherwise create new figure.
    #**kwargs: Additional keyword arguments passed to matplotlib plot().
    x = np.linspace(x_range[0], x_range[1], num_points)
    try:
        # Provide numpy and x in eval namespace
        y = eval(equation_str, {"np": np, "x": x})
    except Exception as e:
        raise ValueError(f"Error parsing equation: {e}")
    
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(x, y, **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'y = {equation_str}')
    return ax


def plot_parametric(x_func_str, y_func_str, t_range=(0, 2*np.pi), num_points=1000, ax=None, **kwargs):
        #x_func_str (str): x(t) as a Python-valid string, e.g., "np.cos(t)"
        #y_func_str (str): y(t) as a Python-valid string, e.g., "np.sin(t)"
        #t_range (tuple): Start and stop for parameter t
        #num_points (int): Number of t values to generate
        #ax (matplotlib.axes._axes.Axes or None): An axis to plot on, or None to create a new figure
        #**kwargs: Additional arguments to matplotlib's plot
    t = np.linspace(t_range[0], t_range[1], num_points)
    try:
        x = eval(x_func_str, {"np": np, "t": t})
        y = eval(y_func_str, {"np": np, "t": t})
    except Exception as e:
        raise ValueError(f"Error parsing equation: {e}")

    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(x, y, **kwargs)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    ax.set_title(f'x(t) = {x_func_str},  y(t) = {y_func_str}')
    return ax


def plot_vector(vector_strs, ax=None, origin=(0, 0), **kwargs):
    """
    Plots a 2D vector given as a list of strings ["x_comp", "y_comp"].

    Parameters:
        vector_strs (list of str): [x-component, y-component] as Python-evaluable strings, e.g., ["3", "2"] or ["np.cos(np.pi/4)", "np.sin(np.pi/4)"]
        ax (matplotlib.axes._axes.Axes or None): Existing axes or None for new plot.
        origin (tuple): Starting point of the vector (default (0, 0))
        **kwargs: Additional arguments passed to matplotlib's quiver.
    
    Returns:
        matplotlib.axes._axes.Axes: The axes object with the plot
    """
    try:
        x_comp = float(eval(vector_strs[0], {"np": np}))
        y_comp = float(eval(vector_strs[1], {"np": np}))
    except Exception as e:
        raise ValueError(f"Error parsing vector components: {e}")

    if ax is None:
        fig, ax = plt.subplots()
    
    ax.quiver(origin[0], origin[1], x_comp, y_comp, angles='xy', scale_units='xy', scale=1, **kwargs)
    ax.set_xlim(min(origin[0], origin[0] + x_comp) - 1, max(origin[0], origin[0] + x_comp) + 1)
    ax.set_ylim(min(origin[1], origin[1] + y_comp) - 1, max(origin[1], origin[1] + y_comp) + 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'2D Vector [{vector_strs[0]}, {vector_strs[1]}]')
    return ax


def plot_parametric_3d(x_func_str, y_func_str, z_func_str, t_range=(0, 2*np.pi), num_points=1000, ax=None, **kwargs):
    """
    Plots a 3D parametric function defined as x(t), y(t), z(t) where all are strings.

    Parameters:
        x_func_str (str): x(t) as a Python-valid string, e.g., "np.cos(t)"
        y_func_str (str): y(t) as a Python-valid string, e.g., "np.sin(t)"
        z_func_str (str): z(t) as a Python-valid string, e.g., "t"
        t_range (tuple): Range for t parameter (default (0, 2*pi))
        num_points (int): Number of t values (default 1000)
        ax (mpl_toolkits.mplot3d.Axes3D or None): Plot on existing axes or create new
        **kwargs: Additional keyword args to pass to ax.plot

    Returns:
        mpl_toolkits.mplot3d.Axes3D: The axes object with the plot
    """

    t = np.linspace(t_range[0], t_range[1], num_points)
    try:
        x = eval(x_func_str, {"np": np, "t": t})
        y = eval(y_func_str, {"np": np, "t": t})
        z = eval(z_func_str, {"np": np, "t": t})
    except Exception as e:
        raise ValueError(f"Error parsing equation: {e}")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, **kwargs)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    ax.set_zlabel('z(t)')
    ax.set_title(f"x(t)={x_func_str}, y(t)={y_func_str}, z(t)={z_func_str}")
    return ax


def plot_surface(equation_str, x_range=(-5, 5), y_range=(-5, 5), num_points=100, ax=None, **kwargs):
    """
    Plots the 3D surface defined by z = f(x, y) where f is a string expression.

    Parameters:
        equation_str (str): Equation in terms of 'x' and 'y', e.g., "np.sin(np.sqrt(x**2 + y**2))"
        x_range (tuple): Range for x values
        y_range (tuple): Range for y values
        num_points (int): Resolution of the meshgrid (default 100)
        ax (mpl_toolkits.mplot3d.Axes3D or None): Axes to plot on, or None for a new plot
        **kwargs: Additional arguments to pass to ax.plot_surface

    Returns:
        mpl_toolkits.mplot3d.Axes3D: The axes object with the plot
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    try:
        Z = eval(equation_str, {"np": np, "x": X, "y": Y})
    except Exception as e:
        raise ValueError(f"Error parsing equation: {e}")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f"z = {equation_str}")
    return ax


def plot_vector_3d(vector_strs, ax=None, origin=(0, 0, 0), **kwargs):
    """
    Plots a 3D vector given as a list of strings ["x_comp", "y_comp", "z_comp"].

    Parameters:
        vector_strs (list of str): [x, y, z] components as strings, e.g. ["1", "2", "3"] or ["np.cos(np.pi/4)", "np.sin(np.pi/4)", "3"]
        ax (mpl_toolkits.mplot3d.Axes3D or None): Plot on existing axes or create new
        origin (tuple): Starting point of the vector (default (0, 0, 0))
        **kwargs: Additional keyword arguments passed to matplotlib's quiver

    Returns:
        mpl_toolkits.mplot3d.Axes3D: The axes object with the plot
    """
    try:
        x_comp = float(eval(vector_strs[0], {"np": np}))
        y_comp = float(eval(vector_strs[1], {"np": np}))
        z_comp = float(eval(vector_strs[2], {"np": np}))
    except Exception as e:
        raise ValueError(f"Error parsing vector components: {e}")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.quiver(
        origin[0], origin[1], origin[2],
        x_comp, y_comp, z_comp,
        length=1, normalize=False, **kwargs
    )

    x_max = max(origin[0], origin[0] + x_comp)
    x_min = min(origin[0], origin[0] + x_comp)
    y_max = max(origin[1], origin[1] + y_comp)
    y_min = min(origin[1], origin[1] + y_comp)
    z_max = max(origin[2], origin[2] + z_comp)
    z_min = min(origin[2], origin[2] + z_comp)
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_zlim(z_min - 1, z_max + 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f'3D Vector [{vector_strs[0]}, {vector_strs[1]}, {vector_strs[2]}]')
    return ax


def plot_contour(equation_str, x_range=(-5, 5), y_range=(-5, 5), num_points=200, ax=None, **kwargs):
    """
    Plots the contour lines of the 3D surface defined by z = f(x, y) as a 2D plot.

    Parameters:
        equation_str (str): Equation in terms of 'x' and 'y', e.g., "np.sin(np.sqrt(x**2 + y**2))"
        x_range (tuple): Range for x values (default (-5, 5))
        y_range (tuple): Range for y values (default (-5, 5))
        num_points (int): Resolution of grid (default 200)
        ax (matplotlib.axes.Axes or None): Axes to plot on, or None for a new plot
        **kwargs: Additional kwargs for matplotlib's contour/contourf

    Returns:
        matplotlib.axes.Axes: The axes object with the plot
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    try:
        Z = eval(equation_str, {"np": np, "x": X, "y": Y})
    except Exception as e:
        raise ValueError(f"Error parsing equation: {e}")

    if ax is None:
        fig, ax = plt.subplots()

    # 'contourf' for filled, 'contour' for lines
    contour = ax.contour(X, Y, Z, **kwargs)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"Contours of z = {equation_str}")
    return ax