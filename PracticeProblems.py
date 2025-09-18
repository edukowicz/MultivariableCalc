import MultivariableCalc as calc
import math

"Page 802, Problem #11"
u = [3, 0, -1]
v = [1, -1, 2]
w = [0, 3, 0]

a = calc.vector_operation(w, v, "subtraction")
print(a)
b = calc.vector_operation(calc.scalar_multiply(u, 6), calc.scalar_multiply(w, 4), "subtraction")
print(b)
c = calc.vector_operation(calc.scalar_multiply(v, -1), calc.scalar_multiply(w, 2), "subtraction")
print(c)
d = calc.scalar_multiply(calc.vector_operation(calc.scalar_multiply(u, 3), v, "addition"), 4)
print(d)
e = calc.vector_operation(calc.scalar_multiply(calc.vector_operation(v, w, "addition"), -8), calc.scalar_multiply(u, 2), "addition")
print(e)
f = calc.vector_operation(calc.scalar_multiply(w, 3), calc.vector_operation(v, w, "subtraction"), "subtraction")
print(f)

"Page 802, Problem #17"

a = calc.unit_vector([-1, 4])
print(a)
b = calc.scalar_multiply(calc.unit_vector([6, -4, 2]), -1)
print(b)
c = calc.unit_vector(calc.vector_operation([3,1,1], [-1, 0, 2], "subtraction"))
print(c)





"Page 874, Problem #11"
r =[]
s =[]
r.append(calc.derivative_as_string("x", "x"))
r.append(calc.derivative_as_string("x**2", "x"))
for i in range(len(r)):
    s.append(calc.evaluate_expression_at_value(r[i], 2))
print(s)

"Page 874, Problem #13"
r =[]
s =[]
r.append(calc.derivative_as_string("sec(x)", "x"))
r.append(calc.derivative_as_string("tan(x)", "x"))
for i in range(len(r)):
    s.append(calc.evaluate_expression_at_value(r[i], 0))
print(s)

"Page 874, Problem #15"
r =[]
s =[]
r.append(calc.derivative_as_string("2*sin(x)", "x"))
r.append(calc.derivative_as_string("1", "x"))
r.append(calc.derivative_as_string("2*cos(x)", "x"))
for i in range(len(r)):
    s.append(calc.evaluate_expression_at_value(r[i], (math.pi)/2))
print(s)

"Page 874, Problem #31"
r =[]
r.append(calc.antiderivative_as_string("3", "x"))
r.append(calc.antiderivative_as_string("4*x", "x"))
print(r)


