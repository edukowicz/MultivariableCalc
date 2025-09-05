import MultivariableCalc as calc

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