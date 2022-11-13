import math
import numpy as np

def calcAngle(a, b ,c):
    # v1 = np.array([Ax - Bx, Ay - By, Az - Bz])
    # v2 = np.array([Cx - Bx, Cy - By, Cz - Bz])
    v1 = np.array([ a[0] - b[0], a[1] - b[1], a[2] - b[2] ])
    v2 = np.array([ c[0] - b[0], c[1] - b[1], c[2] - b[2] ])
    
    v1mag = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
    v1norm = np.array([v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag])

    v2mag = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
    v2norm = np.array([v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag])
    
    res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
    
    angle = np.arccos(res)
    
    return math.degrees(angle)

# a = np.array([100, 100, 80])
# b = np.array([100, 175, 80])
# c = np.array([100, 100, 120])

# print(calcAngle(a, b, c))