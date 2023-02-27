############################################################
# Kim, Myeongkyun, GIST, BioRobotics
# maengkyun@gm.gist.ac.kr
############################################################ 
# Quadrotor Simplified Dynamics & Controller
############################################################
import numpy as np
import matplotlib.pyplot as plt
class Quadrotor():
    def __init__(self, L=0.25,m=1,T=20):
        self.L=L
        self.m=m
        self.A=0.2
        self.g=9.81
        self.I_xx=0.04
        self.I_yy=0.04
        self.I_zz=0.006
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
        
        self.x_data = []
        self.y_data = []
        self.z_data = []
        #self.update_pose(self, x, y, z, roll, pitch, yaw)
    def plot(self,Time):
        #print(Time)
        #print(self.x_data)
        plt.plot(Time,self.x_data)
        plt.plot(Time,self.y_data)
        plt.plot(Time,self.z_data)
        plt.show()
    def Pose_Update(self, Fz,T):
        
        p_ddot_x=(1/self.m)*Fz*(np.sin(self.eta[0])*np.sin(self.eta[2]) + np.sin(self.eta[1])*np.cos(self.eta[0])*np.cos(self.eta[2]))
        p_ddot_y=(1/self.m)*Fz*(-np.sin(self.eta[0])*np.cos(self.eta[2]) + np.sin(self.eta[2])*np.sin(self.eta[1])*np.cos(self.eta[0]))
        p_ddot_z=(1/self.m)*Fz*np.cos(self.eta[0])*np.cos(self.eta[1])-self.g

        eta_ddot_x = (T[0] + T[2]*np.cos(self.eta[0])*np.tan(self.eta[1]) + T[1]*np.sin(self.eta[0])*np.tan(self.eta[1]))/self.I_xx
        eta_ddot_y = (-T[2]*np.sin(self.eta[0]) + T[1]*np.cos(self.eta[0]))/self.I_yy
        eta_ddot_z = (T[2]*np.cos(self.eta[0]) + T[1]*np.sin(self.eta[0]))/(self.I_zz*np.cos(self.eta[1]))

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
        self.x_data.append(self.p[0][0])
        self.y_data.append(self.p[1][0])
        self.z_data.append(self.p[2][0])
        #print(self.x_data)
        #print(self.p)
    def ExtForce(self,f):
        Fz= f[0]+f[1]+f[2]+f[3]
        Tx= 0.707106781186547*self.L*(f[0]-f[1]-f[2]+f[3])
        Ty= 0.707106781186547*self.L*(-f[0]-f[1]+f[2]+f[3])
        Tz= 0.2*(f[0]-f[1] +f[2] -f[3])
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
            throttle = ctlInput[2]+g
            des_Roll = (ctlInput[0]*np.sin(des_Yaw)-ctlInput[1]*np.cos(des_Yaw))/np.sqrt(ctlInput[0]**2+ctlInput[1]**2+(throttle)**2)
            des_Pitch = (ctlInput[0]*np.cos(des_Yaw)+ctlInput[1]*np.sin(des_Yaw))/(throttle)
            
            des_eta = np.array([
                [des_Roll],
                [des_Pitch],
                [des_Yaw]])
            self.Attitude_Controller(throttle, des_eta)
        self.plot(T_data)
        
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
des_Position = np.array([
    [10.0],
    [5.0],
    [5.0]
])
des_Yaw = 0
Q.Position_Controller(des_Position, des_Yaw)
