import MultivariableCalc as calc
import math

#Unit 1.A Practice Problems

""""Page 802, Problem #11"
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
print(r)"""





#Unit 1.B Practice Problems

""""Page 810, Problem #1"
print("Page 810, Problem #1: ")

u = [1, 2]
v = [6, -8]
a = calc.dot_product(u, v)
cosa = a/(calc.vector_magnitude(u)*calc.vector_magnitude(v))
print(a)
print(cosa)

u = [-7, -3]
v = [0, 1]
b = calc.dot_product(u, v)
cosb = b/(calc.vector_magnitude(u)*calc.vector_magnitude(v))
print(b)
print(cosb)

u = [1, -3, 7]
v = [8, -2, -2]
c = calc.dot_product(u, v)
cosc = c/(calc.vector_magnitude(u)*calc.vector_magnitude(v))
print(c)
print(cosc)

u = [-3, 1, 2]
v = [4, 2, -5]
d = calc.dot_product(u, v)
cosd = d/(calc.vector_magnitude(u)*calc.vector_magnitude(v))
print(d)
print(cosd)


"Page 810, Problem #35"
f = [4, -6, 1]
d = calc.scalar_multiply(calc.unit_vector([1, 1, 1]), 15)
work = calc.dot_product(f, d)
print("Page 810, Problem #35: ")
print(work)


"Page 810, Problem #38"
xn = 250 * math.cos(math.radians(38))
yn = 250 * math.sin(math.radians(38))
unknownY = -1 * yn
unknownX = 1000 - xn
force = [unknownX, unknownY]
forceMag = calc.vector_magnitude(force)
angle = math.atan(unknownY/unknownX)
print("Page 810, Problem #38: (magnitdue of force, then angle in radians)")
print(forceMag)
print(angle)


"Page 822, Problem #7"
print("Page 822, Problem #7: ")
u = [2, -1, 3]
v = [0, 1, 7]
w = [1, 4, 5]

a = calc.cross_product(u, calc.cross_product(v, w))
print(a)

b = calc.cross_product(calc.cross_product(u, v), w)
print(b)

c = calc.cross_product(calc.cross_product(u, v), calc.cross_product(v, w))
print(c)

d = calc.cross_product(calc.cross_product(v, w), calc.cross_product(u, v))
print(d)


"Page 822, Problem #10"
u = calc.cross_product([-7, 3, 1], [2, 0, 4])
v = calc.unit_vector(u)
w = calc.scalar_multiply(v, -1)
print("Page 822, Problem #10: ")
print(v)
print(w)


"Page 822, Problem #17"
u = calc.dot_product([2, -3, 1], calc.cross_product([4, 1, -3], [0, 1, 5]))
print("Page 822, Problem #17: ")
print(u)


"Page 822, Problem #37"
f = [200 * math.sin(math.radians(18)), 200 * math.cos(math.radians(18)), 0]
d = [0.2, 0.03, 0]
t = calc.vector_magnitude(calc.cross_product(f, d))
print("Page 822, Problem #37: ")
print(t)


"Page 891, Problem #5"
print("Page 891, Problem #5: ")
r = ["t**2 -1", "t", 0]
tangent = calc.unit_tangent_vector(r)
tangentnum = calc.mv_evaluate_expression_at_value(tangent, 1, "t")
print(tangentnum)
normal = calc.unit_tangent_vector(tangent)
normalnum = calc.mv_evaluate_expression_at_value(normal, 1, "t")
print(normalnum)


"Page 891, Problem #9"
print("Page 891, Problem #9: ")
r = ["4*cos(t)", "4*sin(t)", "t"]
tangent = calc.unit_tangent_vector(r)
tangentnum = calc.mv_evaluate_expression_at_value(tangent, (math.pi)/2, "t")
print(tangentnum)
normal = calc.unit_tangent_vector(tangent)
normalnum = calc.mv_evaluate_expression_at_value(normal, (math.pi)/2, "t")
print(normalnum)


"Page 891, Problem #11"
print("Page 891, Problem #11: ")
r = ["cos(t)*e**t", "sin(t)*e**t", "e**t"]
tangent = calc.unit_tangent_vector(r)
tangentnum = calc.mv_evaluate_expression_at_value(tangent, 0, "t")
print(tangentnum)
print()
normal = calc.unit_tangent_vector(tangent)
normalnum = calc.mv_evaluate_expression_at_value(normal, 0, "t")
print(normalnum)
print()

"Page 891, Problem #15"
print("Page 891, Problem #15: ")
r = ["3*sin(t)", "3*cos(t)", "4*t"]
tangent = calc.unit_tangent_vector(r)
normal = calc.unit_tangent_vector(tangent)
binormal = calc.cross_product_str_to_str(tangent, normal)
print(binormal)
print()


"Page 891, Problem #17"
print("Page 891, Problem #17: ")
r = ["sin(t) - t*cos(t)", "cos(t) + t*sin(t)", "1"]
tangent = calc.unit_tangent_vector(r)
normal = calc.unit_tangent_vector(tangent)
binormal = calc.cross_product_str_to_str(tangent, normal)
print(binormal) """


