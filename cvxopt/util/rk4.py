def rk4(f,x,u,t,dt):
    k1 = f(t,x,u)
    k2 = f(t,x+dt*k1/2,u)
    k3 = f(t,x+dt*k2/2,u)
    k4 = f(t,x+dt*k3,u)
    return x+(k1+2*k2+2*k3+k4)*dt/6