import sys
import time
import rospy
import os
import numpy as np
from scipy.signal import savgol_filter
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from scipy.spatial.transform import Rotation as R
from scipy.io import savemat

## Global variables system
xd = 3.0
yd = -4.6
zd = 5.16
vxd = 0.0
vyd = 0.0
vzd = 0.0

qx = 0.0005
qy = 0.0
qz = 0.0
qw = 1.0
wxd = 0.0
wyd = 0.0
wzd = 0.0



def get_quaternios():
    quat = np.array([qw, qx, qy, qz], dtype=np.double)
    return quat

def get_euler():
    quat = np.array([qx, qy, qz, qw], dtype=np.double)
    rot = R.from_quat(quat)
    eul = rot.as_euler('xyz', degrees=False)

    euler = np.array([eul[0], eul[1], eul[2]], dtype=np.double)
    return euler

def get_euler_p(omega, euler):
    W = np.array([[1, np.sin(euler[0])*np.tan(euler[1]), np.cos(euler[0])*np.tan(euler[1])],
                  [0, np.cos(euler[0]), np.sin(euler[0])],
                  [0, np.sin(euler[0])/np.cos(euler[1]), np.cos(euler[0])/np.cos(euler[1])]])

    euler_p = np.dot(W, omega)
    return euler_p

def get_pos():
    h = np.array([xd, yd, zd], dtype=np.double)
    return h

def get_inertial_vel():
    v = np.array([vxd, vyd, vzd], dtype=np.double)
    return v

def get_body_vel():
    quat = np.array([qx, qy, qz, qw], dtype=np.double)
    rot = R.from_quat(quat)
    rot = rot.as_matrix()
    rot_inv = np.linalg.inv(rot)
    vel_w = np.array([vxd, vyd, vzd], dtype=np.double)
    vel_w = vel_w.reshape(3,1)
    vel = rot_inv@vel_w 

    u = np.array([vel[0,0], vel[1,0], vel[2,0]], dtype=np.double)
    return u

def get_omega():
    omega = np.array([wxd, wyd, wzd], dtype=np.double)
    return omega
# Get system velocities


## Reference system
def send_reference(ref,ref_pub, ref_msg):
        ref_msg.twist.linear.x = ref[0]
        ref_msg.twist.linear.y = ref[1]
        ref_msg.twist.linear.z = ref[2]

        ref_msg.twist.angular.x = 0
        ref_msg.twist.angular.y = 0
        ref_msg.twist.angular.z = ref[3]


        # Publish control values
        ref_pub.publish(ref_msg)
        

def odometry_call_back(odom_msg):
    global xd, yd, zd, qx, qy, qz, qw, vxd, vyd, vzd, wxd, wyd, wzd, time_message

    # Read desired linear velocities from node
    time_message = odom_msg.header.stamp
    xd = odom_msg.pose.pose.position.x 
    yd = odom_msg.pose.pose.position.y
    zd = odom_msg.pose.pose.position.z
    vxd = odom_msg.twist.twist.linear.x
    vyd = odom_msg.twist.twist.linear.y
    vzd = odom_msg.twist.twist.linear.z


    qx = odom_msg.pose.pose.orientation.x
    qy = odom_msg.pose.pose.orientation.y
    qz = odom_msg.pose.pose.orientation.z
    qw = odom_msg.pose.pose.orientation.w

    wxd = odom_msg.twist.twist.angular.x
    wyd = odom_msg.twist.twist.angular.y
    wzd = odom_msg.twist.twist.angular.z
    return None




# Función para limitar la amplitud máxima
def limit_amplitude(signal, max_amplitude):
    peak_value = np.max(np.abs(signal))
    if peak_value > max_amplitude:
        signal *= max_amplitude / peak_value
    return signal

# Función para suavizar la señal
def smooth_signal(signal):
    return savgol_filter(signal, window_length=51, polyorder=3)

# Generar señales combinadas con armónicos y entradas de escalón
def generate_signal(time, index):
    signal = np.zeros_like(time)
    if index == 0:
        signal += 1.7 * np.cos(0.5 * time) + 0.2 * np.sin(0.4 * time)
    elif index == 1:
        signal += 1.4 * np.cos(2 * time) - 0.2 * np.cos(5 * time)
    elif index == 2:
        signal += 1.3 * np.cos(2.5 * time) + 0.2 * np.sin(1.2 * time)
    elif index == 3:
        signal += 1.5 * np.sin(0.8 * time) + 0.3 * np.cos(1.5 * time)
    signal[(time >= 2) & (time < 5)] += 0.3  # Entrada de escalón en 2 <= t < 5 segundos
    signal[(time >= 10) & (time < 15)] += 0.3  # Entrada de escalón en 10 <= t < 15 segundos
    signal[(time >= 7) & (time < 9)] = 0  # Amplitud cero en 7 <= t < 9 segundos
    signal[(time >= 12) & (time < 14)] = 0  # Amplitud cero en 12 <= t < 14 segundos
    signal[(time >= 17) & (time < 19)] = 0  # Amplitud cero en 17 <= t < 19 segundos
    return signal


def main(control_pub, ref_msg):

    # Twist 
    ref_drone = TwistStamped()

    # Simulation time parameters
    ts = 1/30
    tf = 30
    t = np.arange(0, tf+ts, ts, dtype=np.double)

    # States System pose
    states = np.zeros((22, t.shape[0]+1), dtype=np.double)

        ## BOdy velocity
    h = np.zeros((3, t.shape[0]+1), dtype=np.double)
    euler = np.zeros((3, t.shape[0]+1), dtype=np.double) 
    v = np.zeros((3, t.shape[0]+1), dtype=np.double)
    euler_p = np.zeros((3, t.shape[0]+1), dtype=np.double) 
    omega = np.zeros((3, t.shape[0]+1), dtype=np.double)
    quat = np.zeros((4, t.shape[0]+1), dtype=np.double)
    u = np.zeros((3, t.shape[0]+1), dtype=np.double)
    


    # Control signals
    u_ref = np.zeros((4, t.shape[0]), dtype=np.double)

    # Define Control Action

    experiment_number = 10
    
    # Vamos a evaluar diferentes rangos de puntaje
    if experiment_number == 1:
        print("Experimento: 1")
        u_ref[0, :]=  2.4*np.cos(1*t)+0.3*np.sin(0.4*t)
        u_ref[1, :] = 2.2*np.cos(t)*np.cos(0.5*t)+0.25*np.sin(0.7*t)
        u_ref[2, :] = 2.5*np.sin(t)*np.cos(0.5*t) 
        u_ref[3, :] = 1.5*np.sin(t)*np.sin(0.5*t)+0.3*np.cos(0.7*t)*np.cos(0.3*t)
    elif experiment_number == 2:
        print("Experimento: 2")
        u_ref[0, :]=  1.7*np.cos(0.5*t)+0.2*np.sin(0.4*t)
        u_ref[1, :] = 1.1*np.cos(1*t)+0.1*np.cos(2*t)
        u_ref[2, :] = 1.1*np.cos(2*t)+0.1*np.cos(5*t) 
        u_ref[3, :] = 1*np.sin(2*t)
    elif experiment_number == 3:
        print("Experimento: 3")
        u_ref[0, :]=  1.5*(0.7*np.cos(1*t)+0.5*np.sin(3*t))
        u_ref[1, :] = 1.6*(0.7*np.cos(2*t)+0.2*np.sin(0.4*t))
        u_ref[2, :] = 1.1*np.cos(2*t)+0.1*np.cos(5*t) 
        u_ref[3, :] = 0.1*np.sin(2*t) #(0.25 *np.cos(2*t)+ 0.25 *np.sin(0.4*t))
    elif experiment_number == 4:
        print("Experimento: 4")
        val = 2
        u_ref[0, :]= val* 0.9*np.sin(0.5*t)*np.cos(0.5*t)+0.15*np.cos(0.5*t)
        u_ref[1, :] =val* 0.4*np.cos(t)+0.3*np.cos(0.5*t)
        u_ref[2, :] =val* 0.1*np.sin(0.6*t)*np.sin(0.5*t)+0.3*np.cos(0.7*t)*np.cos(0.4*t)
        u_ref[3, :] =val* 0.3*np.sin(0.2*t)*np.cos(0.4*t)-0.15*np.cos(t)
    elif experiment_number == 5:
        print("Experimento: 5")
        u_ref[0, :]=  1.7*np.cos(0.5*t)+0.2*np.sin(0.4*t)
        u_ref[1, :] = 1.1*np.cos(1*t)+0.1*np.cos(2*t)
        u_ref[2, :] = 1.1*np.cos(1*t)+0.1*np.cos(2*t) 
        u_ref[3, :] = 1*np.sin(2*t)
    elif experiment_number == 6:
        print("Experimento: 6")
        u_ref[0, :]=  1.5*np.sin(1*t)*np.cos(0.1*t)-0.15*np.cos(1*t)
        u_ref[1, :] = 1.4*np.cos(t)+0.3*np.cos(0.9*t)
        u_ref[2, :] = 1.5*np.sin(0.75*t)*np.cos(0.1*t)-0.15*np.cos(t) 
        u_ref[3, :] = 1.25 *np.cos(0.5*t)+ 0.25 *np.sin(0.4*t)
    elif experiment_number == 10:
        for i in range(4):
            signal = generate_signal(t, i)
            if i in [1, 2]:  # Aplicar suavizado solo a las señales 2 y 3
                u_ref[i, :] = smooth_signal(limit_amplitude(signal, 10))
            else:
                u_ref[i, :] = limit_amplitude(signal, 10)
                
    else:
        print("Sin experimeto")
        u_ref[0, :]=  0
        u_ref[1, :] = 0
        u_ref[2, :] = 0
        u_ref[3, :] = 0
    


    for k in range(0, 100):
        tic = time.time()
        
        while (time.time() - tic <= ts):
                None
        
        # Save Data
        h[:, 0] = get_pos()
        euler[:, 0] = get_euler()
        v[:, 0] = get_inertial_vel()      
        omega[:, 0] = get_omega()
        euler_p[:, 0] = get_euler_p(omega[:, 0],euler[:, 0])
        quat[:, 0] = get_quaternios()
        u[:, 0] = get_body_vel()

        print("Initializing the experiment")
    
    for k in range(0, t.shape[0]):
        tic = time.time()
        
        
        # Send Control action to the system
        #u_ref[1:4, k]=  [0,0,0]
        send_reference(u_ref[:, k], control_pub, ref_msg)




        # Loop_rate.sleep()
        while (time.time() - tic <= ts):
                None
        toc = time.time() - tic 
        print(toc)

        # Save Data
        h[:, k+1] = get_pos()
        euler[:, k+1] = get_euler()
        v[:, k+1] = get_inertial_vel()     
        omega[:, k+1] = get_omega()
        euler_p[:, k+1] = get_euler_p(omega[:, k+1],euler[:, k+1])
        quat[:, k+1] = get_quaternios()
        u[:, k+1] = get_body_vel()

        states[:, k+1] = np.concatenate((h[:, k+1], euler[:, k+1], v[:, k+1], euler_p[:, k+1], omega[:, k+1], quat[:, k+1], u[:, k+1]))
 




    send_reference([0, 0, 0, 0], control_pub, ref_msg)
    

        

    states_data = {"states": states, "label": "states"}
    u_ref_data = {"u_ref": u_ref, "label": "states_ref"}
    t_data = {"t": t, "label": "time"}


    # FOR IDENTIFICATIONS
    #pwd= "/home/bryansgue/Doctoral_Research/Matlab/Identificacion_M100/IdentificacionAlgoritmos/Ident_Full_model_compact"

    #savemat(os.path.join(pwd, "states_" + str(experiment_number) + ".mat"), states_data)
    #savemat(os.path.join(pwd, "u_ref_" + str(experiment_number) + ".mat"), u_ref_data)
    #savemat(os.path.join(pwd,"t_"+ str(experiment_number) + ".mat"), t_data)

    #For MODEL TESTS
    # Ruta que deseas verificar
    pwd = "/home/bryansgue/Doctoral_Research/Matlab/Graficas_Metologia"

    # Verificar si la ruta no existe
    if not os.path.exists(pwd) or not os.path.isdir(pwd):
        print(f"La ruta {pwd} no existe. Estableciendo la ruta local como pwd.")
        pwd = os.getcwd()  # Establece la ruta local como pwd
    


    Test = "Real"

    if Test == "Mil":
        savemat(os.path.join(pwd, "states_Mil_" + str(experiment_number) + ".mat"), states_data)
        savemat(os.path.join(pwd, "u_ref_Mil_" + str(experiment_number) + ".mat"), u_ref_data)
        savemat(os.path.join(pwd,"t_Mil_"+ str(experiment_number) + ".mat"), t_data)
    elif Test == "Hil":
        savemat(os.path.join(pwd, "states_Hil_" + str(experiment_number) + ".mat"), states_data)
        savemat(os.path.join(pwd, "u_ref_Hil_" + str(experiment_number) + ".mat"), u_ref_data)
        savemat(os.path.join(pwd,"t_Hil_"+ str(experiment_number) + ".mat"), t_data)  
    elif Test == "Real":
        savemat(os.path.join(pwd, "states_Real_" + str(experiment_number) + ".mat"), states_data)
        savemat(os.path.join(pwd, "u_ref_Real_" + str(experiment_number) + ".mat"), u_ref_data)
        savemat(os.path.join(pwd,"t_Real_"+ str(experiment_number) + ".mat"), t_data) 

    else:
        print("Sin experimeto")
        u_ref[0, :]=  0
        u_ref[1, :] = 0
        u_ref[2, :] = 0
        u_ref[3, :] = 0

    


    return None


if __name__ == '__main__':
    try:
        # Initialization Node
        rospy.init_node("Python_Node",disable_signals=True, anonymous=True)

        # Odometry topic
        odometry_webots = "/dji_sdk/odometry"
        odometry_subscriber = rospy.Subscriber(odometry_webots, Odometry, odometry_call_back)

        # Cmd Vel topic
        velocity_topic = "/m100/velocityControl"
        velocity_message = TwistStamped()
        velocity_publisher = rospy.Publisher(velocity_topic, TwistStamped, queue_size = 10)

        main(velocity_publisher, velocity_message)



    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("Error System")
        send_reference([0, 0, 0, 0], velocity_publisher, velocity_message)
        pass
    else:
        print("Complete Execution")
        send_reference([0, 0, 0, 0], velocity_publisher, velocity_message)
        pass