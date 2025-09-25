import MultivariableCalc as mvcalc
import math
import sympy as sp

#part a 

 #state given quantities from problem statement 

#set variable "rampdist" to the length of the incline from where the skiier jumped to where he landed
rampdist = 259 
#set variable "ramppangle" to the angle between the incline and the horizontal
rampangle = (-1)*(math.radians(23))
#set the intitial displacement vector of the skiier: both the x- and y-components are equal to zero
xi = [0,0]
#set the final displacement vector of the skiier (after he lands): the x-component is equal to rampdist*math.cos(rampangle) and the y-component is equal to rampdist*math.sin(rampangle)
xf = [rampdist*math.cos(rampangle),rampdist*math.sin(rampangle)] 
#set the acceleration vector of the skiier: there is no acceleration in the x direction, and the y-component of acceleration is equal to -32 ft/s^2

#get initial expressions for x- and y-velocities and positions 

v = mvcalc.mv_function_integral(a, ["t"], "t") #using my calculator function 
v[0] = str(v[0]) + " + v_0x" 
v[1] = str(v[1]) + " + v_0y" 

x = mvcalc.mv_function_integral(v, ["t"], "t") #using my calculator function 
x[0] = str(x[0]) + " + " + str(xi[0])
x[1] = str(x[1]) + " + " + str(xi[1]) 

 
#solving for v_0x and v_0y 

t = sp.symbols('t') 

x_expr = sp.sympify(x[0]) 
result_x = x_expr.subs(t, 2.9) 
solution_x = sp.solve(sp.Eq(result_x, xf[0]), 'v_0x') 
v0x = solution_x[0] 
v[0] = v[0].replace("v_0x", str(v0x)) 

y_expr = sp.sympify(x[1]) 
result_y = y_expr.subs(t, 2.9) 
solution_y = sp.solve(sp.Eq(result_y, xf[1]), 'v_0y') 
v0y = solution_y[0]  
v[1] = v[1].replace("v_0y", str(v0y)) 


#final solution to problem 

v0 = math.sqrt(v0x**2 + v0y**2) 
angle = math.degrees(math.atan(v0y/v0x)) 
print("v0 =",v0,"angle =",angle) 


#part b 

#finding the total distance travelled 

speed = "sqrt((" + str(v[0]) + ")**2 + (" + str(v[1]) + ")**2)" 
d = mvcalc.mv_function_definite_integral([speed], ["t"], "t", 0, 2.9) #using my calculator function 
print("The distance travelled is",d[0]) 

 