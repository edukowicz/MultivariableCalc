import MultivariableCalc as calc
import math

#define important known vectors
r = [2615, 15881, 3980]
v = [-2.767, -0.7905, 4.980]
i = [1, 0, 0]
j = [0, 1, 0]
k = [0, 0, 1]

#find vector h
hvec = calc.cross_product(r, v)
h = calc.vector_magnitude(hvec)
print(f"h = {h} km^2/s")
print()

#find vector e
mew = 398600
runit = calc.unit_vector(r)
evec = calc.scalar_multiply(calc.vector_operation(calc.cross_product(v, hvec), calc.scalar_multiply(runit, mew), "subtraction"),(1/mew))
e = calc.vector_magnitude(evec)
print(f"e = {e}")
print()

#find vector for node line
node = calc.cross_product(k, hvec)

#find angle i
eqplane = calc.cross_product(node, k)
oplane = calc.cross_product(node, hvec)
eqdoto = calc.dot_product(eqplane, oplane)
eqmag = calc.vector_magnitude(eqplane)
omag = calc.vector_magnitude(oplane)
irad = math.acos(eqdoto/(eqmag*omag))
iangle = math.degrees(irad)
print(f"i = {iangle} degrees")
print()


#find angle omega
nodedoti = calc.dot_product(node, i)
nodemag = calc.vector_magnitude(node)
omegarad = math.acos(nodedoti/nodemag)
omega = math.degrees(omegarad)
print(f"omega = {omega} degrees")
print()

#find angle theta
rdote = calc.dot_product(r, evec)
rmag = calc.vector_magnitude(r)
emag = calc.vector_magnitude(evec)
thetarad = math.acos(rdote/(rmag*emag))
theta = math.degrees(thetarad)
print(f"theta = {theta} degrees")
print()

#find angle w
nodedote = calc.dot_product(node, evec)
wrad = math.acos(nodedote/(emag*nodemag))
w = math.degrees(wrad)
print(f"w = {w} degrees")