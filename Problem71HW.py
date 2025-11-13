"""
import MultivariableCalc as mvcalc
import math
import sympy as sp

#part a 

 #state given quantities from problem statement 

#set variable "rampdist" to the length of the incline from where the skier jumped to where he landed
rampdist = 259 
#set variable "ramppangle" to the angle between the incline and the horizontal
rampangle = (-1)*(math.radians(23))
#set the variable "xi" to the intitial displacement vector of the skier: both the x- and y-components are equal to zero ft
xi = [0,0]
#set the variable "xf" to the final displacement vector of the skier (after he lands): the x-component is equal to rampdist*math.cos(rampangle) ft and the y-component is equal to rampdist*math.sin(rampangle) ft
xf = [rampdist*math.cos(rampangle),rampdist*math.sin(rampangle)] 
#set the acceleration vector of the skier: there is no acceleration in the x direction, and the y-component of acceleration is equal to -32 ft/s^2
a = ["0", "-32"]

#get initial expressions for x- and y-velocities and positions 

#set the variable "v" to the expression for the velocity vector of the skier, found by taking the indefinite integral of the skier's acceleration vector
v = mvcalc.mv_function_integral(a, ["t"], "t") #using my calculator function 
#add "v_0x" to the x-component of the velocity vector, which represents the x-component of velocity at time t = 0 (which is the constant of integration -- still unsolved)
v[0] = str(v[0]) + " + v_0x"
#add "v_0y" to the y-component of the velocity vector, which represents the y-component of velocity at time t = 0 (which is the constant of integration -- still unsolved)
v[1] = str(v[1]) + " + v_0y"

#set the variable "x" to the expression for the displacement vector of the skier, found by taking the indefinite integral of the skier's velocity vector
x = mvcalc.mv_function_integral(v, ["t"], "t") #using my calculator function 
#add xi[0] to the x-component of the displacement vector, which is the inital displacement in the x-direction of the skier (this step "solves" the initial condition)
x[0] = str(x[0]) + " + " + str(xi[0])
#add xi[1] to the y-component of the displacement vector, which is the inital displacement in the y-direction of the skier (this step "solves" the initial condition)
x[1] = str(x[1]) + " + " + str(xi[1]) 

 
#solving for v_0x and v_0y 

#create a symbolic variable "t" that represents the time since the skier has jumped from the ramp
t = sp.symbols('t') 

#set the variable "x_expr" to the expression for x-component of the displacement vector of the skier converted to a SymPy expression
x_expr = sp.sympify(x[0]) 
#set the variable "result_x" to the x-component of the displacement vector when the skier lands 2.9 seconds after jumping (this is done by plugging in 2.9 for t in the x-component of the displacement function)
result_x = x_expr.subs(t, 2.9) 
#set the variable "solution_x" to the solved value of "v_0x" (the intial velocity in the x-direction) -- do this by setting "result_x" (expression for x-component of displacement vector when skier lands, v_0x is only unknown) equal to "xf[0]" (value for x-component of displacement vector when skier lands) and solving for "v_0x"
solution_x = sp.solve(sp.Eq(result_x, xf[0]), 'v_0x') 
#set the variable "v0x" equal to "solution_x[0]", the initial value of the x-component of velocity 
v0x = solution_x[0] 
#in the expression for the x-component of velocity, replace "v_0x" with "v0x" (the value of the inital x-component of velocity) -- solved the initial condition
v[0] = v[0].replace("v_0x", str(v0x)) 

#set the variable "y_expr" to the expression for y-component of the displacement vector of the skier converted to a SymPy expression
y_expr = sp.sympify(x[1]) 
#set the variable "result_y" to the y-component of the displacement vector when the skier lands 2.9 seconds after jumping (this is done by plugging in 2.9 for t in the y-component of the displacement function)
result_y = y_expr.subs(t, 2.9)
#set the variable "solution_y" to the solved value of "v_0y" (the intial velocity in the y-direction) -- do this by setting "result_y" (expression for y-component of displacement vector when skier lands, v_0y is only unknown) equal to "xf[1]" (value for y-component of displacement vector when skier lands) and solving for "v_0y"
solution_y = sp.solve(sp.Eq(result_y, xf[1]), 'v_0y')
#set the variable "v0y" equal to "solution_y[0]", the initial value of the y-component of velocity 
v0y = solution_y[0] 
#in the expression for the y-component of velocity, replace "v_0y" with "v0y" (the value of the inital y-component of velocity) -- solved the initial condition
v[1] = v[1].replace("v_0y", str(v0y)) 


#final solution to problem 

#set the inital speed of the skier "v0" to the magnitude of the skier's initial velocity (using components "v0x" and "v0y")
v0 = math.sqrt(v0x**2 + v0y**2) 
#set the angle between the jump and the horizonal equal to arctan(v0y/v0x)
angle = math.degrees(math.atan(v0y/v0x)) 
#print the intial speed of the skier and the angle between the jump and the horizontal
print("v0 =",v0,"angle =",angle) 


#part b 

#finding the total distance travelled 

#create a string "speed" that is a function representing the skier's speed at a time t after they jump from the ramp
speed = "sqrt((" + str(v[0]) + ")**2 + (" + str(v[1]) + ")**2)" 
#find the total distance travelled ("d") by the skier by taking the definite interal of the function for the skiers speed ("speed") over the time the skier was in the air -- the equation for the arc length of a function
#the limits of integration are 0 and 2.9, which are the times that the skier leaves the ramp and lands on the incline, respectively
d = mvcalc.mv_function_definite_integral([speed], ["t"], "t", 0, 2.9) #using my calculator function 
#print the total distance travelled by the skier while in the air
print("The distance travelled is",d[0])  
"""