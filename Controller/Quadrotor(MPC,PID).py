
############################################################
# Kim, Myeongkyun, GIST, BioRobotics
# maengkyun@gm.gist.ac.kr
############################################################ 
# Quadrotor Simplified Dynamics
############################################################
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
class Quadrotor():
    def __init__(self, L=0.2,m=1,T=15):
        self.L=L
        self.m=m
        self.A=0.2
        self.g=9.81
        self.I_xx=4e-3
        self.I_yy=4e-3
        self.I_zz=8.4e-3
        self.dt=0.02
        self.Time=T
        
        self.p = np.array([
            [0.0],
            [0.0],
            [0.0]])
        self.p_dot = np.array([
            [0.0],
            [0.0],
            [0.0]])
        self.eta = np.array([
            [0.0],
            [0.0],
            [0.0]])
        self.eta_dot = np.array([
            [0.0],
            [0.0],
            [0.0]])
        self.T = np.array([
            [m, m, m, m],
            [L/np.sqrt(2), -L/np.sqrt(2), -L/np.sqrt(2), L/np.sqrt(2)],
            [-L/np.sqrt(2), -L/np.sqrt(2), L/np.sqrt(2), L/np.sqrt(2)],
            [self.A, -self.A, self.A, -self.A]])
        self.p_ddot = np.array([
            [0.0],
            [0.0],
            [0.0]])
        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.fig, self.ax = plt.subplots()
        #self.update_pose(self, x, y, z, roll, pitch, yaw)
    def plot(self,Time):
        #print(Time)
        #print(self.x_data)
        self.ax.clear()
        self.ax.plot(Time,self.x_data,label='x')
        self.ax.plot(Time,self.y_data,label='y')
        self.ax.plot(Time,self.z_data,label='z')
        plt.legend()
        plt.pause(0.01)
    def Pose_Update(self, Fz,T):
        
        p_ddot_x=(1/self.m)*Fz*(np.sin(self.eta[0])*np.sin(self.eta[2]) + np.sin(self.eta[1])*np.cos(self.eta[0])*np.cos(self.eta[2]))
        p_ddot_y=(1/self.m)*Fz*(-np.sin(self.eta[0])*np.cos(self.eta[2]) + np.sin(self.eta[2])*np.sin(self.eta[1])*np.cos(self.eta[0]))
        p_ddot_z=(1/self.m)*Fz*np.cos(self.eta[0])*np.cos(self.eta[1])-self.g
        self.p_ddot[0] = p_ddot_x
        self.p_ddot[1] = p_ddot_y
        self.p_ddot[2] = p_ddot_z
        eta_ddot_x = (T[0] + T[2]*np.cos(self.eta[0])*np.tan(self.eta[1]) + T[1]*np.sin(self.eta[0])*np.tan(self.eta[1]))/self.I_xx
        eta_ddot_y = (-T[2]*np.sin(self.eta[0]) + T[1]*np.cos(self.eta[0]))/self.I_yy
        eta_ddot_z = (T[2]*np.cos(self.eta[0]) + T[1]*np.sin(self.eta[0]))/(self.I_zz*np.cos(self.eta[1]))
        #print(p_ddot_z)

        self.p_dot[0] = self.p_dot[0] + p_ddot_x*self.dt
        self.p_dot[1] = self.p_dot[1] + p_ddot_y*self.dt
        self.p_dot[2] = self.p_dot[2] + p_ddot_z*self.dt

        self.eta_dot[0] = self.eta_dot[0] + eta_ddot_x*self.dt
        self.eta_dot[1] = self.eta_dot[1] + eta_ddot_y*self.dt
        self.eta_dot[2] = self.eta_dot[2] + eta_ddot_z*self.dt

        self.p[0]=self.p[0] + self.p_dot[0]*self.dt
        self.p[1]=self.p[1] + self.p_dot[1]*self.dt
        self.p[2]=self.p[2] + self.p_dot[2]*self.dt

        self.eta[0] = self.eta[0] + eta_ddot_x*self.dt
        self.eta[1] = self.eta[1] + eta_ddot_y*self.dt
        self.eta[2] = self.eta[2] + eta_ddot_z*self.dt
        #self.R = np.matrix([ [np.cos(self.eta[2])*np.cos(self.eta[1]),     np.sin(self.eta[0])*np.sin(self.eta[1])*np.cos(self.eta[2])-np.sin(self.eta[2])*np.cos(self.eta[0]),  np.sin(self.eta[0])*np.sin(self.eta[2])+np.sin(self.eta[1])*np.cos(self.eta[0])*np.cos(self.eta[2])],
        #        [np.sin(self.eta[2])*np.cos(self.eta[1]),     np.sin(self.eta[0])*np.sin(self.eta[2])*np.sin(self.eta[1])+np.cos(self.eta[0])*np.cos(self.eta[2]), -np.sin(self.eta[0])*np.cos(self.eta[2])+np.sin(self.eta[2])*np.sin(self.eta[1])*np.cos(self.eta[0])],
        #        [            np.sin(self.eta[1]),                             np.sin(self.eta[0])*np.sin(self.eta[1])*np.cos(self.eta[2]),                                      np.cos(self.eta[0])*np.cos(self.eta[1])]
        #        ])
        
        #self.x_data.append(self.p[0][0])
        #self.y_data.append(self.p[1][0])
        #self.z_data.append(self.p[2][0])
        #print(self.x_data)
        #print(self.p)
    def ExtForce(self,f):
        Fz= f[0]+f[1]+f[2]+f[3]
        Tx= 0.707106781186547*self.L*(f[0]-f[1]-f[2]+f[3])
        Ty= 0.707106781186547*self.L*(-f[0]-f[1]+f[2]+f[3])
        Tz= 0.2*(f[0]-f[1] +f[2] -f[3])
        #print(f[0],f[1],f[2],f[3])
        Torque=np.array([
            [Tx],
            [Ty],
            [Tz]])
        self.Pose_Update(Fz,Torque)

    def T_Inverse(self,Fz,T):
        F=np.array([
            [Fz],
            [T[0]],
            [T[1]],
            [T[2]]])
        F=F.reshape((4,1))
        #print(Fz)
        #print(self.T.shape, F.shape)
        f=np.linalg.inv(self.T)@F
        self.ExtForce(f)

        #f1=0.25*Fz + 0.353553390593274*T[0]/self.L - 0.353553390593274*T[1]/self.L + 0.25*T[2]/self.A
        #f2=0.25*Fz - 0.353553390593274*T[0]/self.L - 0.353553390593274*T[1]/self.L - 0.25*T[2]/self.A
        #f3=0.25*Fz - 0.353553390593274*T[0]/self.L + 0.353553390593274*T[1]/self.L + 0.25*T[2]/self.A
        #f4=0.25*Fz + 0.353553390593274*T[0]/self.L + 0.353553390593274*T[1]/self.L - 0.25*T[2]/self.A
    def Body_Conv(self,p_ddot_d,eta_ddot):
        v_dot_d=(p_ddot_d)*np.cos(self.eta[0])*np.cos(self.eta[1])
        torque_roll = self.I_xx*eta_ddot[0] - self.I_xx*eta_ddot[2]*np.sin(self.eta[1])
        torque_pitch = self.I_yy*eta_ddot[2]*np.sin(self.eta[0])*np.cos(self.eta[1]) + self.I_yy*eta_ddot[1]*np.cos(self.eta[0])
        torque_yaw = self.I_zz*eta_ddot[2]*np.cos(self.eta[0])*np.cos(self.eta[1]) - self.I_zz*eta_ddot[1]*np.sin(self.eta[0])
        T=np.array([
            [torque_roll],
            [torque_pitch],
            [torque_yaw]
        ])
        self.T_Inverse(v_dot_d,T)
    # def getPosition(self):
    #     return self.p
    # def getAngle(self):
    #     return self.eta
    # def getVelociT[1](self):
    #     return self.p_dot
    # def getAngularVelociT[1](self):
    #     return self.eta_dot

        
    def Attitude_Controller(self,throttle, des_eta):
        eta_Kp = 10.0
        eta_Kp_rate = 3.0
        #print(self.eta.shape, des_eta.shape)
        etaError = des_eta-self.eta
        ctlInput = eta_Kp*etaError - eta_Kp_rate*self.eta_dot
        #print(throttle)
        self.Body_Conv(throttle,ctlInput)

    def Position_Controller(self,des_Position,des_Yaw):
        pos_Kp = 1.0
        pos_Kp_rate = 0.8
        g=9.81
        T_data=[]
        for time in np.linspace(0,self.Time,self.Time*int(1/self.dt)):
            T_data.append(time)
            posError = des_Position-self.p
            ctlInput = saturation(pos_Kp*posError - pos_Kp_rate*self.p_dot,-3,3)
            throttle = ctlInput[2]+self.m*g
            des_Roll = (ctlInput[0]*np.sin(des_Yaw)-ctlInput[1]*np.cos(des_Yaw))/np.sqrt(ctlInput[0]**2+ctlInput[1]**2+(throttle)**2)
            des_Pitch = (ctlInput[0]*np.cos(des_Yaw)+ctlInput[1]*np.sin(des_Yaw))/(throttle)
            
            des_eta = np.array([
                [des_Roll],
                [des_Pitch],
                [des_Yaw]])
            self.Attitude_Controller(throttle, des_eta)
        self.plot(T_data)
    def MPC_Controller(self,x0,des_Position):
        A = np.array([
                    [1, 0,self.dt, 0],
                    [0, 1, 0, self.dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                    ])
        B = np.array([
                    [0, 0],
                    [0, 0],
                    [0, 1],
                    [-1, 0]
        ])

        # MPC parameters
        N = 10 # Number of prediction steps
        Nc = 10 # Number of ctl prediction steps
        Q = np.diag([3.0, 3.0 , 1.0, 1.0]) # State cost matrix
        R = np.diag([0.1, 0.1]) # Input cost matrix
        P = Q # terminal state cost
        # Define initial state and reference
        

        x_ref = np.zeros((4,1))
        x_ref[:2]=des_Position[:2]
        # Define obstacle positions and radii
        torque = np.zeros((3,1))
        u_max=np.pi/9.0
        u_min=-np.pi/9.0

            # Define decision variables
        x = cp.Variable((4, N+1)) # State
        u = cp.Variable((2, N)) # Input
        #print(self.Time*int(1/self.dt))
        # Define cost function
        cost = 0
        constraints = []
        constraints = [x[:,0] == x0[:,0]]
        #constraints = [x[:,0] == x_ref[:,0]]
        for k in range(N):
            cost += cp.quad_form(x[:,k]-x_ref[:,0], Q)+cp.quad_form(u[:,k], R)
            #if(k<Nc-1):
                #constraints += [cp.norm(u[:,k+1]-u[:,k],'inf') <= 0.3]
        cost += cp.quad_form(x[:,N]-x_ref[:,0], P)
        # Define constraints
        
        for k in range(N):
            constraints += [x[:,k+1] == A @ x[:,k] + B @ u[:,k]]
            #constraints += [cpx[1] <= 3] # Input constraints
            constraints += [u[:,k] <= u_max] # Input constraints
            constraints += [u[:,k] >= u_min] # Input constraints
            #constraints += [cp.abs(x[1,k]) <= 2] # Input constraints
            #constraints += [cp.abs(x[2,k]) <= 1] # Input constraints
        # Solve optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver='OSQP')
        torque[:2] = np.array(u.value[:,0]).reshape(2,1)
        #print('torque',torque)
        self.Attitude_Controller(self.m*self.g,torque)
        
        x0 = np.array([
            [self.p[0][0]],
            [self.p[1][0]],
            [self.p_dot[0][0]],
            [self.p_dot[1][0]]])
        self.x_data.append(self.p[0][0])
        self.y_data.append(self.p[1][0])
        return x0
    def Altitude_Controller(self,x0,des_Altitude):
        A = np.array([
                    [1,self.dt],
                    [0, 1]
                    ])
        B = np.array([
                    [0],
                    [1/self.m]
        ])

        # MPC parameters
        N = 10 # Number of prediction steps
        Nc = 10 # Number of ctl prediction steps
        Q = np.diag([30.0, 15.0]) # State cost matrix
        R = np.diag([0.1]) # Input cost matrix
        P = Q # terminal state cost
        # Define initial state and reference
        
        torque = np.zeros((3,1))
        x_ref = np.zeros((2,1))
        x_ref[0]=des_Position[2]
        
        u_max=3
        u_min=-3
        #x_ref[1] = saturation(x_ref[0] - x0[0],-3,3) # desired velosity
        #x_ref[1]=saturation(x_ref[1])
        # Define decision variables
        x = cp.Variable((2, N+1)) # State
        u = cp.Variable((1, N)) # Input
        #print(self.Time*int(1/self.dt))
        # Define cost function
        cost = 0
        constraints = []
        constraints = [x[:,0] == x0[:,0]]
        #constraints = [x[:,0] == x_ref[:,0]]
        for k in range(N):
            cost += cp.quad_form(x[:,k]-x_ref[:,0], Q)+cp.quad_form(u[:,k], R)
            #if(k<Nc-1):
                #constraints += [cp.norm(u[:,k+1]-u[:,k],'inf') <= 0.3]
        cost += cp.quad_form(x[:,N]-x_ref[:,0], P)
        # Define constraints
        
        for k in range(N):
            constraints += [x[:,k+1] == A @ x[:,k] + B @ u[:,k]]
            #constraints += [cpx[1] <= 3] # Input constraints
            #constraints += [u[:,k] <= u_max] # Input constraints
            #constraints += [u[:,k] >= u_min] # Input constraints
            #constraints += [cp.abs(x[1,k]) <= 2] # Input constraints
            #constraints += [cp.abs(x[2,k]) <= 1] # Input constraints
        # Solve optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver='OSQP')
        throttle = u.value[0,0]+self.m*self.g
        self.Body_Conv(throttle,torque)
        
        x0 = np.array([
            [self.p[2][0]],
            [self.p_dot[2][0]]])
        self.z_data.append(self.p[2][0])


        return x0
    
def saturation(input, min,max):
    satur=[]
    for data in input:
        if data<min:
            data=min
        elif data>max:
            data=max
        satur.append(data)
    
    input = np.array(satur)
    return input
Q=Quadrotor()
state_xy = np.array([
    [0],
    [0],
    [0],
    [0]
    ])
state_z = np.array([
    [0],
    [0]
    ])
des_Yaw = 0
#Q.Position_Controller(des_Position, des_Yaw)

des_Position = np.array([
    [5.0],
    [5.0],
    [10.0]
])
T_data=[]
for time in np.linspace(0,Q.Time,int(Q.Time/Q.dt)):
    #x_ref[1] = saturation(x_ref[0] - x0[0],-3,3) # desired velosity
    #x_ref[1]=saturation(x_ref[1])
    T_data.append(time)
    state_xy = Q.MPC_Controller(state_xy,des_Position)
    state_z = Q.Altitude_Controller(state_z,des_Position)
    #print(state_xy[0],state_xy[1],state_z[0])
    Q.plot(T_data)