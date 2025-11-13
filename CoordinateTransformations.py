"""
import MultivariableCalc as calc
import math
import ClassicalElements

r = [4623.458, 5533.656, 4646.733]
v = [-6.033, 3.485, 4.480]

#find vector h
h = ClassicalElements.magH(r, v)
print(f"h = {h} km^2/s")
print()

#find vector e
e = ClassicalElements.magE(r, v)
print(f"e = {e}")
print()

#find angle i
iangle = ClassicalElements.angleI(r, v)
print(f"i = {iangle} degrees")
print()

#find angle omega
omega = ClassicalElements.angleOmega(r, v)
print(f"omega = {omega} degrees")
print()

#find angle theta
theta = ClassicalElements.angleTheta(r, v)
print(f"theta = {theta} degrees")
print()

#find angle w
wangle = ClassicalElements.angleW(r, v)
print(f"w = {wangle} degrees")
print()

#coordinate transformations

o = math.radians(omega)
i = math.radians(iangle)
w = math.radians(wangle)

rijk = [[4623.458], [5533.656], [4646.733]]
vijk = [[-6.033], [3.485], [4.480]]

rotationMatrix = [
    [math.cos(w)*math.cos(o)-math.sin(o)*math.sin(w)*math.cos(i), math.cos(w)*math.sin(o)+math.sin(w)*math.cos(i)*math.cos(o), math.sin(w)*math.sin(i)],
    [-1*math.sin(w)*math.cos(o)-math.sin(o)*math.cos(w)*math.cos(i), -1*math.sin(w)*math.sin(o)+math.cos(o)*math.cos(w)*math.cos(i), math.cos(w)*math.sin(i)],
    [math.sin(o)*math.sin(i), -1*math.sin(i)*math.cos(o), math.cos(i)]
] 

rpqw = calc.matrix_product(rotationMatrix, rijk)
vpqw = calc.matrix_product(rotationMatrix, vijk)

print(f"r in the perifocal (pqw) frame = {rpqw}")
print(f"v in the perifocal (pqw) frame = {vpqw}")
"""
