import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import torch

import cvxopt

class Env:
    def __init__(self,headless=True):
        self.headless = headless
        if self.headless:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            self.client = bc.BulletClient(connection_mode=p.GUI)
            self.client.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        self.goal_vel = 1
        self.dt = 0.01

    def reset(self):
        self.setup()
        self.get_ob()

        return self.ob

    def setup(self):
        self.client.resetSimulation()
        self.client.setTimeStep(self.dt)
        self.client.setGravity(0,0,-9.8)
        self.client.setPhysicsEngineParameter(enableConeFriction=0)

        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = self.client.loadURDF('plane.urdf',useFixedBase=True)
        self.client.changeDynamics(self.plane,-1,lateralFriction=0,spinningFriction=0,rollingFriction=0)

        self.box_collision = self.client.createCollisionShape(p.GEOM_BOX,halfExtents=[1,1,1])
        self.box_visual = self.client.createVisualShape(p.GEOM_BOX,halfExtents=[1,1,1],rgbaColor=[0,0,1,1])

        self.robot = self.client.createMultiBody(
            baseMass=1., baseCollisionShapeIndex=self.box_collision,
            baseVisualShapeIndex=self.box_visual, basePosition=[0,0,1],
            flags=p.URDF_ENABLE_SLEEPING|p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.client.changeDynamics(self.robot,-1,lateralFriction=0,spinningFriction=0,rollingFriction=0,linearDamping=0)
        self.mass = 1


    def step(self,u):
        u *= self.mass
        self.client.applyExternalForce(self.robot,-1,[u[0],u[1],0],[0,0,0],p.LINK_FRAME)
        #self.client.applyExternalTorque(self.robot,-1,[0,0,math.atan2(self.ob[3],self.ob[2])-self.ori[2]],p.LINK_FRAME)

        self.get_ob()
        self.client.stepSimulation()

        return self.ob

    def close(self):
        self.client.disconnect()

    def get_ob(self):
        pose,oriq = self.client.getBasePositionAndOrientation(self.robot)
        self.ori = p.getEulerFromQuaternion(oriq)
        vel,avel = self.client.getBaseVelocity(self.robot)
        #self.ob = torch.tensor([pose[0],pose[1],ori[2],vel[0],vel[1],avel[2]]).view(-1,1)
        self.ob = torch.tensor([pose[0],pose[1],vel[0],vel[1]]).view(-1,1)

if __name__ == '__main__':
    import time
    import math

    ## initialize the environment
    env = Env(headless=False)
    ob = env.reset()

    ## double integrator of dim n
    n = 2

    ## continuous state space
    A = torch.zeros(2*n,2*n)
    A[:n,n:] = torch.eye(n)
    B = torch.zeros(2*n,n)
    B[n:,:] = torch.eye(n)

    ## discrete state space
    T = 0.01 #sampling time
    Ad = torch.eye(2*n) + A * T
    Bd = B * T

    ## lqr params
    Q = torch.diag(torch.tensor([100000,100000,1,1]))
    R = 1
    N = 0
    H = 0.1
    dt = T

    LQR = cvxopt.LQRsolver(Ad,Bd,Q,R,N,H,dt)
    K = LQR.get_K()

    print('pose: {}'.format(ob.T))
    steps = 200
    rad = 5
    goals = [torch.tensor([rad*math.cos(2*math.pi*i/steps)-rad,rad*math.sin(2*math.pi*i/steps),math.sin(2*math.pi*i/steps),math.cos(2*math.pi*i/steps)]).view(-1,1) for i in range(steps+1)]
    #goals = [torch.tensor([0,0,0,2]).view(-1,1) for _ in range(100)]

    for goal in goals:
        print('goal: {}'.format(goal.T))
        for i in range(K.size(0)):
            u_ = -K[i,...]@(ob-goal)
            ob = env.step(u_)
            if not env.headless: time.sleep(1/100)
        print('pose: {}'.format(ob.T))
    env.close()

