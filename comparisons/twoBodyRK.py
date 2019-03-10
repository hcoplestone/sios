from numpy import *
from matplotlib.pyplot import *
# planet configurations
# RS: radius of orbit in AU
# Ks: mass as ratio of sun (here actually earth mass, but will normalize just below)
# Vs: initial speed modifier to get ellipeses
# Ps: plot style for the planet
# sun ; mercu ; venus ; earth ; mars ; jupi ; satu ;
Rs = array([ 0.0 , 0.4 , 0.7 , 1.0 , 1.5 , 5.2 , 9.5 ])
Ks = array([ 332946.0 , 0.055 , 0.815 , 1.0 , 0.107 , 318 , 95.0 ])
Vs = array([ 0.0 , 1.05 , 1.02 , 1.02 , 1.02 , 1.02 , 1.02 ])
Ps = array([ 'xr' , '-g' , '-g' , '-b' , '-r' , '-k' , '-k' ])
# normalize mass to solar mass
Ks = Ks / Ks[0]
# initial value settings
#InitState = zeros((5*size(Rs),1))
InitState = zeros(5*size(Rs))
for i in range(0,size(Rs)):
    if Rs[i]==0: # special "sun" support
        initial_speed = 0
    else:
        initial_speed = (2*pi*sqrt((Ks[0]+Ks[i])/Rs[i]))*Vs[i]
        InitState[i*5+0] = Rs[i] # x
        InitState[i*5+1] = 0.0 # vx
        InitState[i*5+2] = 0.0 # y
        InitState[i*5+3] = initial_speed # vy
        InitState[i*5+4] = Ks[i] # mass

# Runge-Kutta integrator of order 4
def RK4(generate_ds, tspan, InitState, timestepsize):
    # set step size
    h = timestepsize # renaming to short name
    # allocate memory
    estimatedcols = int(floor((tspan[1]-tspan[0]) / h) + 1)
    States = zeros((size(InitState),estimatedcols))
    Times = zeros(estimatedcols)
    idx = 0
    # set the initial state
    States[:,0] = InitState
    Times[0] = tspan[0]
    State = InitState;
    for t in arange(tspan[0],tspan[1],h):
        # calculate new a value based on fourth order Runge-Kutta
        k1 = h * generate_ds(t, State)
        k2 = h * generate_ds(t+h/2, State+k1/2)
        k3 = h * generate_ds(t+h/2, State+k2/2)
        k4 = h * generate_ds(t+h, State+k3)
        State = State + (k1+2*k2+2*k3+k4)/6
        # save the state
        idx = idx+1
        States[:,idx] = State
        Times[idx] = t+h
    return Times,States

# set up the differential, function for usage in Runge-Kutta
def dState(t, S):
    # allocating memory
    dS = zeros(size(S))
    # loop through planets to be updated
    for i in range(0,int(size(S)/5)):
        # speed
        dS[i*5+0] = S[i*5+1]
        dS[i*5+2] = S[i*5+3]
        # prep for acceleration calc
        x = S[i*5+0]
        y = S[i*5+2]
        # loop through planets affecting the current planet
        for j in range(0,int(size(S)/5)):
            if j!=i:
                pk = S[j*5+4]
                px = S[j*5+0]
                py = S[j*5+2]
                d = sqrt((px-x)**2 + (py-y)**2)
                # acceleration
                dS[i*5+1] = dS[i*5+1] - 4*(pi**2)*pk * (x - px)/(d**3)
                dS[i*5+3] = dS[i*5+3] - 4*(pi**2)*pk * (y - py)/(d**3)
    return dS

# the time span to run the simulation in years
tspan = [ 0 , 5 ]
timestepsize = 0.001
# call the integrator
Times,States = RK4(dState, tspan, InitState, timestepsize)
# plot planetary orbit
figure(1)
for i in range(0,size(Rs)):
    xidx = i*5+0
    yidx = i*5+2
    plot(States[xidx,:], States[yidx,:], Ps[i], ms=1)
title('Planet orbits')
xlabel('x')
ylabel('y')
xlim(-12,12)
ylim(-12,12)