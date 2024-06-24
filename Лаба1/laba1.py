import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math as math
import matplotlib as matplotlib
from matplotlib.animation import FuncAnimation

matplotlib.use("TkAgg")

v0 = 10
R = 5
t = sp.Symbol('t')

rx = v0*t-R*sp.cos(v0*t/R-math.pi/2)
ry = R + R*sp.sin(v0*t/R-math.pi/2)

vpx = sp.diff(rx, t)
vpy = sp.diff(ry, t)
vp = sp.sqrt(vpx*vpx+vpy*vpy)

wpx = sp.diff(vpx, t)
wpy = sp.diff(vpy, t)

Wt = sp.diff(vp, t)
W = sp.sqrt(wpx*wpx+wpy*wpy)
Wn = (abs(W*W-Wt*Wt))**0.5

T = np.linspace(0, 20, 2001)
xn = np.zeros_like(T)
yn = np.zeros_like(T)
vx = np.zeros_like(T)
vy = np.zeros_like(T)
v = np.zeros_like(T)
wt = np.zeros_like(T)
wn = np.zeros_like(T)



for i in range(len(T)):
    xn[i] = sp.Subs(rx, t, T[i])
    yn[i] = sp.Subs(ry, t, T[i])
    vx[i] = sp.Subs(vpx, t, T[i])
    vy[i] = sp.Subs(vpy, t, T[i])
    v[i] = sp.Subs(vp, t, T[i])
    wt[i] = sp.Subs(Wt, t, T[i])
    wn[i] = sp.Subs(Wn, t, T[i])


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.plot(xn, yn)

P = ax.plot(xn[0], yn[0], marker='o')[0]

Phi = math.atan2(vy[0], vx[0])

VLine = ax.plot([xn[0], xn[0]+vx[0]], [yn[0], yn[0]+vy[0]], 'black')[0]

def rotate(x, y, a):
    x_rotated = x * np.cos(a) - y * np.sin(a)
    y_rotated = x * np.sin(a) + y * np.cos(a)
    return x_rotated, y_rotated

V_arrow_x = np.array([-v[0]*0.1, 0.0, -v[0]*0.1], dtype=float)
V_arrow_y = np.array([v[0]*0.05, 0.0, -v[0]*0.05], dtype=float)
V_arrow_rotx, V_arrow_roty = rotate(V_arrow_x, V_arrow_y, Phi)
V_arrow, = ax.plot(xn[0] + vx[0] + V_arrow_rotx, yn[0] + vy[0] + V_arrow_roty, color="black")

WTLine = ax.plot([xn[0], xn[0]+wt[0]*math.cos(Phi)], [yn[0], yn[0]+wt[0]*math.sin(Phi)], 'red')[0]

WT_arrow_x = np.array([-wt[0]*0.1, 0.0, -wt[0]*0.1], dtype=float)
WT_arrow_y = np.array([wt[0]*0.05, 0.0, -wt[0]*0.05], dtype=float)
WT_arrow_rotx, WT_arrow_roty = rotate(WT_arrow_x, WT_arrow_y, Phi)
WT_arrow, = ax.plot(xn[0]+wt[0]*math.cos(Phi) + WT_arrow_rotx, yn[0]+wt[0]*math.sin(Phi) + WT_arrow_roty, color="red")


WNLine = ax.plot([xn[0], xn[0]+wn[0]*math.cos(Phi-math.pi/2)], [yn[0], yn[0]+wn[0]*math.sin(Phi-math.pi/2)], 'blue')[0]

WN_arrow_x = np.array([-wn[0]*0.1, 0.0, -wn[0]*0.1], dtype=float)
WN_arrow_y = np.array([wn[0]*0.05, 0.0, -wn[0]*0.05], dtype=float)
WN_arrow_rotx, WN_arrow_roty = rotate(WN_arrow_x, WN_arrow_y, Phi-math.pi/2)
WN_arrow, = ax.plot(xn[0]+wn[0]*math.cos(Phi-math.pi/2) + WN_arrow_rotx, yn[0]+wn[0]*math.sin(Phi-math.pi/2) + WN_arrow_roty, color="blue")



def cha(i):
    P.set_data(xn[i], yn[i])

    Phi = math.atan2(vy[i], vx[i])

    VLine.set_data([xn[i], xn[i]+vx[i]], [yn[i], yn[i]+vy[i]])

    V_arrow_x = np.array([-v[i]*0.1, 0.0, -v[i]*0.1], dtype=float)
    V_arrow_y = np.array([v[i]*0.05, 0.0, -v[i]*0.05], dtype=float)
    V_arrow_rotx, V_arrow_roty = rotate(V_arrow_x, V_arrow_y, Phi)
    V_arrow.set_data(xn[i] + vx[i] + V_arrow_rotx, yn[i] + vy[i] + V_arrow_roty)

    WTLine.set_data([xn[i], xn[i] + wt[i] * math.cos(Phi)], [yn[i], yn[i] + wt[i] * math.sin(Phi)])

    WT_arrow_x = np.array([-wt[i]*0.1, 0.0, -wt[i]*0.1], dtype=float)
    WT_arrow_y = np.array([wt[i]*0.05, 0.0, -wt[i]*0.05], dtype=float)
    WT_arrow_rotx, WT_arrow_roty = rotate(WT_arrow_x, WT_arrow_y, Phi)
    WT_arrow.set_data(xn[i] + wt[i] * math.cos(Phi) + WT_arrow_rotx, yn[i] + wt[i] * math.sin(Phi) + WT_arrow_roty)

    WNLine.set_data([xn[i], xn[i] + wn[i] * math.cos(Phi-math.pi/2)], [yn[i], yn[i] + wn[i] * math.sin(Phi-math.pi/2)])

    WN_arrow_x = np.array([-wn[i]*0.1, 0.0, -wn[i]*0.1], dtype=float)
    WN_arrow_y = np.array([wn[i]*0.05, 0.0, -wn[i]*0.05], dtype=float)
    WN_arrow_rotx, WN_arrow_roty = rotate(WN_arrow_x, WN_arrow_y, Phi - math.pi / 2)
    WN_arrow.set_data(xn[i] + wn[i] * math.cos(Phi - math.pi / 2) + WN_arrow_rotx, yn[i] + wn[i] * math.sin(Phi - math.pi / 2) + WN_arrow_roty)

    return [P]

a = FuncAnimation(fig, cha, frames=len(T), interval=10)
plt.show()
