"""
import MultivariableCalc as calc
import math

def vecH(r, v):
    hvec = calc.cross_product(r, v)
    return hvec

def magH(r, v):
    h = calc.vector_magnitude(vecH(r, v))
    return h

def vecE(r, v):
    mew = 398600
    runit = calc.unit_vector(r)
    hvec = vecH(r, v)
    evec = calc.scalar_multiply(calc.vector_operation(calc.cross_product(v, hvec), calc.scalar_multiply(runit, mew), "subtraction"),(1/mew))
    return evec

def magE(r, v):
    e = calc.vector_magnitude(vecE(r, v))
    return e

def node(r, v):
    hvec = vecH(r, v)
    k = [0, 0, 1]
    nodeLine = calc.cross_product(k, hvec)
    return nodeLine

def angleI(r, v):
    nodeLine = node(r, v)
    hvec = vecH(r, v) 
    k = [0, 0, 1]
    eqplane = calc.cross_product(nodeLine, k)
    oplane = calc.cross_product(nodeLine, hvec)
    eqdoto = calc.dot_product(eqplane, oplane)
    eqmag = calc.vector_magnitude(eqplane)
    omag = calc.vector_magnitude(oplane)
    irad = math.acos(eqdoto/(eqmag*omag))
    iangle = math.degrees(irad)
    return iangle

def angleOmega(r, v):
    nodeLine = node(r, v)
    i = [1, 0, 0]
    nodedoti = calc.dot_product(nodeLine, i)
    nodemag = calc.vector_magnitude(nodeLine)  
    omegarad = math.acos(nodedoti/nodemag)
    omega = math.degrees(omegarad)
    return omega

def angleTheta(r, v):
    evec = vecE(r, v)
    rdote = calc.dot_product(r, evec)
    rmag = calc.vector_magnitude(r)
    emag = calc.vector_magnitude(evec)
    thetarad = math.acos(rdote/(rmag*emag))
    theta = math.degrees(thetarad)
    return theta

def angleW(r, v):
    nodeLine = node(r, v)
    evec = vecE(r, v)
    nodedote = calc.dot_product(nodeLine, evec)
    nodemag = calc.vector_magnitude(nodeLine)
    emag = calc.vector_magnitude(evec)
    wrad = math.acos(nodedote/(emag*nodemag))
    w = math.degrees(wrad)
    return w
"""    