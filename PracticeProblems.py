import MultivariableCalc as calc
import math

"Page 802, Problem #11"
u = [3, 0, -1]
v = [1, -1, 2]
w = [0, 3, 0]
a = calc.vector_operation(w, v, "subtraction")
print("Page 802, Problem #11a: ")
print(a)
b = calc.vector_operation(calc.scalar_multiply(u, 6), calc.scalar_multiply(w, 4), "addition")
print("Page 802, Problem #11b: ")
print(b)
c = calc.vector_operation(calc.scalar_multiply(v, -1), calc.scalar_multiply(w, 2), "subtraction")
print("Page 802, Problem #11c: ")
print(c)
d = calc.scalar_multiply(calc.vector_operation(calc.scalar_multiply(u, 3), v, "addition"), 4)
print("Page 802, Problem #11d: ")
print(d)
e = calc.vector_operation(calc.scalar_multiply(calc.vector_operation(v, w, "addition"), -8), calc.scalar_multiply(u, 2), "addition")
print("Page 802, Problem #11e: ")
print(e)
f = calc.vector_operation(calc.scalar_multiply(w, 3), calc.vector_operation(v, w, "subtraction"), "subtraction")
print("Page 802, Problem #11f: ")
print(f)


"Page 802, Problem #17"
a = calc.unit_vector([-1, 4])
print("Page 802, Problem #17a: ")
print(a)
b = calc.scalar_multiply(calc.unit_vector([6, -4, 2]), -1)
print("Page 802, Problem #17b: ")
print(b)
c = calc.unit_vector(calc.vector_operation([3,1,1], [-1, 0, 2], "subtraction"))
print("Page 802, Problem #17c: ")
print(c)




"Page 874, Problem #11"
r =[]
s =[]
r.append(calc.derivative_as_string("x", "x"))
r.append(calc.derivative_as_string("x**2", "x"))
for i in range(len(r)):
    s.append(calc.evaluate_expression_at_value(r[i], 2))
print("Page 874, Problem #11: ")
print(s)

"Page 874, Problem #13"
r =[]
s =[]
r.append(calc.derivative_as_string("sec(x)", "x"))
r.append(calc.derivative_as_string("tan(x)", "x"))
for i in range(len(r)):
    s.append(calc.evaluate_expression_at_value(r[i], 0))
print("Page 874, Problem #13: ")
print(s)

"Page 874, Problem #15"
r =[]
s =[]
r.append(calc.derivative_as_string("2*sin(x)", "x"))
r.append(calc.derivative_as_string("1", "x"))
r.append(calc.derivative_as_string("2*cos(x)", "x"))
for i in range(len(r)):
    s.append(calc.evaluate_expression_at_value(r[i], (math.pi)/2))
print("Page 874, Problem #15: ")
print(s)

"Page 874, Problem #31"
r =[]
r.append(calc.antiderivative_as_string("3", "t"))
r.append(calc.antiderivative_as_string("4*t", "t"))
print("Page 874, Problem #31: ")
print(r)


"Page 874, Problem #35"
r = []
r.append(calc.antiderivative_as_string("t**2", "t"))
r.append(calc.antiderivative_as_string("-2*t", "t"))
r.append(calc.antiderivative_as_string("1/t", "t"))
print("Page 874, Problem #35: ")
print(r)


"Page 874, Problem #37"
r = []
r.append(calc.definite_integral_as_string("cos(2*t)", 0, (math.pi)/2, "t"))
r.append(calc.definite_integral_as_string("sin(2*t)", 0, (math.pi)/2, "t"))
print("Page 874, Problem #37: ")
print(r)


"Page 874, Problem #41"
r = []
r.append(calc.definite_integral_as_string("t**0.5", 1, 9, "t"))
r.append(calc.definite_integral_as_string("t**(-0.5)", 1, 9, "t"))
print("Page 874, Problem #41: ")
print(r)