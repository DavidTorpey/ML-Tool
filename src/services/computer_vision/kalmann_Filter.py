import numpy as np

# This is a 2D example
# We will track postion and velocity
# position,velicity

intial_values = np.array([4000, 280])

observation_values = np.array([[4000, 280],[4260, 282],[4550, 285],[4860, 286],[5110, 290]])


# intial conditions
acceleration = 2
delta_t = 1
velocity = 280
delta_x = 25

A_matrix = np.array([[1, delta_t], [0, 1]])
B_matrix = np.array([[0.5*np.power(delta_t,2)], [delta_t]])

#-----------------Everything below this is generalized-----------------#

number_of_dimensions = observation_values.shape[1]

# Process errors in Process covariance matrix 
delta_P_position = 20
delta_P_velocity = 5
P_errors = np.array([delta_P_position, delta_P_velocity])

# Observation errors
obs_delta_x = 25
obs_delta_velocity = 6
obs_errors = np.array([obs_delta_x, obs_delta_velocity])
init_R_covar_matrix = np.zeros([obs_errors.shape[0], obs_errors.shape[0]])
np.fill_diagonal(init_R_covar_matrix, np.power(obs_errors, 2))

init_P_covar_matrix = np.zeros([P_errors.shape[0], P_errors.shape[0]])
np.fill_diagonal(init_P_covar_matrix, np.power(P_errors, 2))
previous_P_covar = init_P_covar_matrix.copy()

H = np.identity(number_of_dimensions)
C = np.identity(number_of_dimensions)
I = np.identity(number_of_dimensions)
previous_X_k = intial_values

#Begin loop itr here

for i in range(observation_values.shape[0]-1):
# State matrix
# X_kp = AX_k-1 + BU_k + W_k
# X deals with velocity and position
# U deals with acceleration
# W deals with noise (Won't consider)
    AX = np.dot(A_matrix, previous_X_k.transpose())
    BU = np.dot(B_matrix, [acceleration])
    X_kp = AX + BU

# Process Errors in Process Covariance Matrix
# This means we assume there will be some errors in the process when calculating the next state
# Predicted process covariance matrix
# P_k = AP_k-1A_t + Q_k
# Won't consider Q_k
# Non diag elements set to 0 for simplicity
    AP_k = np.dot(A_matrix, previous_P_covar)
    AP_K_A = np.dot(AP_k, A_matrix.transpose())
    AP_K_A_diag = np.diag(AP_K_A)
    P_kp = np.zeros(AP_K_A.shape)
    np.fill_diagonal(P_kp, AP_K_A_diag)
    
# Kalmann Gain
# KG = (P_k*H)/(H*P_K*H_t + R)
# H is to manipulate The calculation to get it in the correct dimentions
# R is for observation errors which might occur 
# if KG is high we trust measurement more. If KG is low, we trust prediction more
    P_k_H = np.dot(P_kp, H.transpose())
    HPH_R = np.dot(H, P_k_H) + init_R_covar_matrix
    KG = np.zeros(init_R_covar_matrix.shape)
    KG_diag = np.diag(P_k_H)/np.diag(HPH_R)
    np.fill_diagonal(KG,KG_diag)
    
# The new observation
# Use this to adjust our next state 
# Y_k = CY_k + Z_k
# C used to get it into the correct dims
    Y_k = np.dot(C,  observation_values[i+1,:])
    
# Calculate the current state
# X_k = X_kp + KG[Y_k - HX_kp]
# X_kp is the predicted value
    X_k = X_kp + np.dot(KG,(Y_k - np.dot(H, X_kp)))

# Update process covar matrix 
# P_k = (I - K*H)* P_kp
# P_kp is the predicted process covar matrix defined above
    I_KG_H = (I - np.dot(KG, H))
    P_k = np.dot(I_KG_H, P_kp)
    
    previous_P_covar = P_k
    previous_X_k = X_k
    
    print previous_P_covar
    print previous_X_k
    print "--------------------------------"
    



