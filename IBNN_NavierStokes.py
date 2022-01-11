# Immersed boundary neural network
# Description: This code solves the immersed boundary problem using a neural
# network approach for a elastic band test problem. The data is obtained using
# the finite difference code IB2D
#
# Author: Ryan Konno
#
# Code was adapted from
#
# M. Raissi, P. Perdikaris, and G. Karniadakis. Physics-informed neural networks:
# A deep learn-ing framework for solving forward and inverse problems involving nonlinear
# partial differential equations. J. Comp. Phys., 378:686–707, 2019. doi:https://doi.org/10.1016/j.jcp.2018.10.045.
#
# and
#
# Reymundo Itzá Balam, Francisco Hernandez-Lopez, Joel Trejo-Sánchez, Miguel Uh Zapata. An
# immersed boundary neural network for solving elliptic equations with singular forces on
# arbitrary domains. Mathematical Biosciences and Engineering, 2021, 18(1): 22-56. doi:
# 10.3934/mbe.2021002

# Install for google collab
!pip install pyDOE

# Access location on drive
from google.colab import drive
drive.mount('/content/drive')
%tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import math

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:

# Initialize the class
def __init__(self, x, y, t, u, v, fx, fy, layers):

    X = np.concatenate([x, y, t], 1)

    # self.lambda_1 = 1000
    # self.lambda_2 = 6.65
    self.lambda_1 = 1
    self.lambda_2 = 0.1

    self.lb = X.min(0)
    self.ub = X.max(0)

    self.X = X

    self.x = X[:,0:1]
    self.y = X[:,1:2]
    self.t = X[:,2:3]

    self.u = u
    self.v = v

    # Set the RHS data
    self.fx_exct = fx
    self.fy_exct = fy

    self.layers = layers

    # Initialize NN
    self.weights, self.biases = self.initialize_NN(layers)

    # tf placeholders and graph
    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                 log_device_placement=True))

    self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
    self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
    self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])

    self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
    self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])

    self.fx_tf = tf.placeholder(tf.float32, shape=[None, self.fx_exct.shape[1]])
    self.fy_tf = tf.placeholder(tf.float32, shape=[None, self.fy_exct.shape[1]])

    self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf, self.t_tf)

    # Define the loss function for the NN
    # Here we us a mean-square loss function
    self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                tf.reduce_mean(tf.square(self.v_tf - self.v_pred)) + \
                tf.reduce_mean(tf.square(self.f_u_pred - self.fx_tf)) + \
                tf.reduce_mean(tf.square(self.f_v_pred - self.fy_tf))

    # Define the L-BFGS-B optimizer
    self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                            method = 'L-BFGS-B',
                                                            options = {'maxiter': 20000,
                                                                       'maxfun': 100000,
                                                                       'maxcor': 50,
                                                                       'maxls': 50,
                                                                       'ftol' : 1.0 * np.finfo(float).eps})

    # Define the Adams optimizer
    # This part of the code was adapted to include a decaying step size to improve convergence
    # Optimizer with decaying step size:
    starter_learning_rate = 0.001
    decay_steps = 100
    # decay_rate = 0.1 # fixed value
    decay_rate = np.exp(decay_steps/50000*(np.log(1e-5)-np.log(starter_learning_rate))) # Value from homework NN
    # global_step = 1
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase=True, name=None)
    print(decay_rate)
    self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate)
    self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, global_step = global_step)

    init = tf.global_variables_initializer()
    self.sess.run(init)

def initialize_NN(self, layers):
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0,num_layers-1):
        W = self.xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)
    return weights, biases

def xavier_init(self, size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

# Compute the neural network
def neural_net(self, X, weights, biases):
    num_layers = len(weights) + 1

    H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

#  Navier-Stokes network
def net_NS(self,x,y,t):
    lambda_1 = self.lambda_1
    lambda_2 = self.lambda_2
    psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
    psi = psi_and_p[:,0:1]
    p = psi_and_p[:,1:2]

    u = tf.gradients(psi, y)[0]
    v = -tf.gradients(psi, x)[0]

    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]

    v_t = tf.gradients(v, t)[0]
    v_x = tf.gradients(v, x)[0]
    v_y = tf.gradients(v, y)[0]
    v_xx = tf.gradients(v_x, x)[0]
    v_yy = tf.gradients(v_y, y)[0]

    p_x = tf.gradients(p, x)[0]
    p_y = tf.gradients(p, y)[0]

    # Function for immersed boundary
    f_u = lambda_1*u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy)
    f_v = lambda_1*v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)

    return u,v,p, f_u, f_v

# Callback to print loss function
def callback(self, loss):
    print('Loss: %.3e' % (loss))

# Function to train the NN
def train(self, nIter):
    tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
              self.u_tf: self.u, self.v_tf: self.v, self.fx_tf: self.fx_exct,
              self.fy_tf: self.fy_exct}

    start_time = time.time()

    # Loop to run the Adams optimizer and then the BFGS optimizer
    for it in range(nIter):
        self.sess.run(self.train_op_Adam, tf_dict)

        # Print
        if it % 100 == 0:
            elapsed = time.time() - start_time
            loss_value = self.sess.run(self.loss, tf_dict)
            print('It: %d, Loss: %.3e, Time: %.2f' %
                  (it, loss_value, elapsed))
            start_time = time.time()

    self.optimizer.minimize(self.sess,
                        feed_dict = tf_dict,
                        fetches = [self.loss],
                        loss_callback = self.callback)

# Function used for testing to predict the solution (u,v,p) at tstar
def predict(self, x_star, y_star, t_star):

    tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}

    u_star = self.sess.run(self.u_pred, tf_dict)
    v_star = self.sess.run(self.v_pred, tf_dict)
    p_star = self.sess.run(self.p_pred, tf_dict)

    return u_star, v_star, p_star

if __name__ == "__main__":

N_train = 125 # Number of training parameters

# Specify layers for NS equations
# Need 3 for input (x,y,t)
# Need 2 for output (psi,p)
width = 80
layers = [3, width, width, width, width, width, width, width, width,  2]

# Load Data
data = scipy.io.loadmat('/content/drive/MyDrive/APMA922Project/rubberband_data_lessdata.mat') # Data generated by IB2D for rubberband simulation
# data = scipy.io.loadmat('/content/drive/MyDrive/APMA922Project/rubberband_data.mat') # Data generated by IB2D for rubberband simulation

# Organize the data
U_star = data['U_star'] # N x 2 x T
P_star = data['p_star'] # N x T
t_star = data['t'] # T x 1
X_star = data['X_star'] # N x 2
F_star = data['F_star'] # N x 2 x T

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data
XX = np.tile(X_star[:,0:1], (1,T)) # N x T
YY = np.tile(X_star[:,1:2], (1,T)) # N x T
TT = np.tile(t_star, (1,N)).T # N x T

UU = U_star[:,0,:] # N x T
VV = U_star[:,1,:] # N x T
PP = P_star # N x T
FX = F_star[:,0,:] # N x T
FY = F_star[:,1,:] # N x T

x = XX.flatten()[:,None] # NT x 1
y = YY.flatten()[:,None] # NT x 1
t = TT.flatten()[:,None] # NT x 1

u = UU.flatten()[:,None] # NT x 1
v = VV.flatten()[:,None] # NT x 1
p = PP.flatten()[:,None] # NT x 1
fx = FX.flatten()[:,None] # NT x 1
fy = FY.flatten()[:,None] # NT x 1

# Organize the Training Data
idx = np.random.choice(N*T, N_train, replace=False)
x_train = x[idx,:]
y_train = y[idx,:]
t_train = t[idx,:]
u_train = u[idx,:]
v_train = v[idx,:]
fx_train = fx[idx,:]
fy_train = fy[idx,:]

# Initialize
model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, fx_train, fy_train, layers)

# Train the NN
start = time.time()
model.train(5000)
end = time.time()
print(end - start)

# Organize Test Data
snap = np.array([25])
x_star = X_star[:,0:1]
y_star = X_star[:,1:2]
t_star = TT[:,snap]

u_star = U_star[:,0,snap]
v_star = U_star[:,1,snap]
p_star = P_star[:,snap]
fx_star = F_star[:,0,snap]
fy_star = F_star[:,1,snap]

# Prediction
start = time.time()
u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
end = time.time()
print(end - start)

# Calculate the L2 error in u,v,p
error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)

print('Error u: %e' % (error_u))
print('Error v: %e' % (error_v))
print('Error p: %e' % (error_p))
