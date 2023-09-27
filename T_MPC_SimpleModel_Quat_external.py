from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template import AcadosModel
import scipy.linalg
import numpy as np
import time
import matplotlib.pyplot as plt
from casadi import Function
from casadi import MX
from casadi import reshape
from casadi import vertcat
from casadi import horzcat
from casadi import cos
from casadi import sin
from casadi import solve
from casadi import inv
from casadi import mtimes

import rospy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
#from c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython
from geometry_msgs.msg import TwistStamped
import math

# CARGA FUNCIONES DEL PROGRAMA
from fancy_plots import plot_pose, plot_error, plot_time
from Functions_SimpleModel import f_system_simple_model_quat
from Functions_SimpleModel import f_d, odometry_call_back, get_odometry_simple_quat, send_velocity_control, pub_odometry_sim_quat
import P_UAV_simple

# Global variables Odometry Drone
x_real = 1
y_real = 1
z_real = 5
vx_real = 0.0
vy_real = 0.0
vz_real = 0.0
qw_real = 1
qx_real = 0
qy_real = 0.0
qz_real = 0
wx_real = 0.0
wy_real = 0.0
wz_real = 0.0






def create_ocp_solver_description(x0, N_horizon, t_horizon, zp_max, zp_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model, f_system = f_system_simple_model_quat()
    ocp.model = model
    ocp.p = model.p
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    # set dimensions
    ocp.dims.N = N_horizon

    Q_mat = MX.zeros(3, 3)
    Q_mat[0, 0] = 1.1
    Q_mat[1, 1] = 1.1
    Q_mat[2, 2] = 1.1

    R_mat = MX.zeros(4, 4)
    R_mat[0, 0] = 1
    R_mat[1, 1] = 1
    R_mat[2, 2] = 1
    R_mat[3, 3] = 1
    
    ocp.parameter_values = np.zeros(ny)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    error_pose = ocp.p[0:3] - model.x[0:3]
    ocp.model.cost_expr_ext_cost = error_pose.T @ Q_mat @error_pose  + model.u.T @ R_mat @ model.u 
    ocp.model.cost_expr_ext_cost_e = error_pose.T @ Q_mat @ error_pose


    # set constraints
    #ocp.constraints.lbu = np.array([-2, -2, -2])
    #ocp.constraints.ubu = np.array([2, 2, 2])
    #ocp.constraints.idxbu = np.array([0, 1, 2])

    ocp.constraints.x0 = x0

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    #ocp.solver_options.tol = 1e-4

    # set prediction horizon
    ocp.solver_options.tf = t_horizon

    return ocp

def main(vel_pub, vel_msg, odom_sim_pub, odom_sim_msg):
    # Initial Values System
    # Simulation Time
    t_final = 60
    # Sample time
    frec= 30
    t_s = 1/frec
    # Prediction Time
    N_horizont = 50
    t_prediction = N_horizont/frec

    # Nodes inside MPC
    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # Sample time vector
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)

    # Vector Initial conditions
    x = np.zeros((11, t.shape[0]+1-N_prediction), dtype = np.double)
   

    # Read Real data
    x[:, 0] = get_odometry_simple_quat()
    x[:, 0] = [1,1,0,1,0,0,0,0,0,0,0]

    #TAREA DESEADA
    num = 4
    xd = lambda t: 4 * np.sin(5*0.04*t) + 3
    yd = lambda t: 4 * np.sin(5*0.08*t)
    zd = lambda t: 2.5 * np.sin (0.2* t) + 5  
    xdp = lambda t: 4 * 5 * 0.04 * np.cos(5*0.04*t)
    ydp = lambda t: 4 * 5 * 0.08 * np.cos(5*0.08*t)
    zdp = lambda t: 2.5 * 0.2 * np.cos(0.2 * t)

    hxd = xd(t)
    hyd = yd(t)
    hzd = zd(t)
    hxdp = xdp(t)
    hydp = ydp(t)
    hzdp = zdp(t)

    psid = np.arctan2(hydp, hxdp)
    psidp = np.gradient(psid, t_s)

    # Reference Signal of the system
    xref = np.zeros((15, t.shape[0]), dtype = np.double)
    xref[0,:] = 5
    xref[1,:] = 5
    xref[2,:] = 5  
    xref[3,:] = 1 
    xref[4,:] = 0
    xref[5,:] = 0 
    xref[6,:] = 0 
    xref[7,:] = 0 
    # Initial Control values
    u_control = np.zeros((4, t.shape[0]-N_prediction), dtype = np.double)
    #u_control = np.zeros((4, t.shape[0]), dtype = np.double)

    # Limits Control values
    zp_ref_max = 3
    phi_max = 3
    theta_max = 3
    psi_max = 2

    zp_ref_min = -zp_ref_max
    phi_min = -phi_max
    theta_min = -theta_max
    psi_min = -psi_max

    # Create Optimal problem
    model, f = f_system_simple_model_quat()

    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, zp_ref_max, zp_ref_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min)
    #acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

    solver_json = 'acados_ocp_' + model.name + '.json'
    AcadosOcpSolver.generate(ocp, json_file=solver_json)
    AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    #acados_ocp_solver = AcadosOcpSolverCython(ocp.model.name, ocp.solver_options.nlp_solver_type, ocp.dims.N)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    simX = np.ndarray((nx, N_prediction+1))
    simU = np.ndarray((nu, N_prediction))

    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(x[:,0].shape))
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # Errors of the system
    Ex = np.zeros((1, t.shape[0]-N_prediction), dtype = np.double)
    Ey = np.zeros((1, t.shape[0]-N_prediction), dtype = np.double)
    Ez = np.zeros((1, t.shape[0]-N_prediction), dtype = np.double)

    # Simulation System
    ros_rate = 30  # Tasa de ROS en Hz
    rate = rospy.Rate(ros_rate)  # Crear un objeto de la clase rospy.Rate

    for k in range(0, t.shape[0]-N_prediction):
        tic = time.time()
        Ex[:, k] = xref[0, k] - x[0, k]
        Ey[:, k] = xref[1, k] - x[1, k]
        Ez[:, k] = xref[2, k] - x[2, k]

        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

        # SET REFERENCES
        for j in range(N_prediction):
            yref = xref[:,k+j]
            acados_ocp_solver.set(j, "p", yref)

        yref_N = xref[:,k+N_prediction]
        acados_ocp_solver.set(N_prediction, "p", yref_N)

        # get solution
        for i in range(N_prediction):
            simX[:,i] = acados_ocp_solver.get(i, "x")
            simU[:,i] = acados_ocp_solver.get(i, "u")
        simX[:,N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        print(simX[:,10])

        u_control[:, k] = simU[:,0]

        # Get Computational Time
        status = acados_ocp_solver.solve()

        toc_solver = time.time()- tic

        # Get Control Signal
        u_control[:, k] = acados_ocp_solver.get(0, "u")

        print(u_control[:, k])
        #u_control[:, k] = [0.1, 0.0, 0.0, 0]
        send_velocity_control(u_control[:, k], vel_pub, vel_msg)

        # System Evolution
        opcion = "Sim"  # Valor que quieres evaluar

        if opcion == "Real":
            x[:, k+1] = get_odometry_simple_quat()
        elif opcion == "Sim":
            x[:, k+1] = f_d(x[:, k], u_control[:, k], t_s, f)
            pub_odometry_sim_quat(x[:, k+1], odom_sim_pub, odom_sim_msg)
        else:
            print("Opción no válida")
        
        
        delta_t[:, k] = toc_solver
        
        print("x:", " ".join("{:.2f}".format(value) for value in np.round(x[0:12, k], decimals=2)))
        
        rate.sleep() 
        toc = time.time() - tic 
        
        
    
    send_velocity_control([0, 0, 0, 0], vel_pub, vel_msg)

    

    fig1, ax11 = fancy_plots_1()
    states_x, = ax11.plot(t[0:x.shape[1]], x[0,:],
                    color='#BB5651', lw=2, ls="-")
    states_y, = ax11.plot(t[0:x.shape[1]], x[1,:],
                    color='#69BB51', lw=2, ls="-")
    states_z, = ax11.plot(t[0:x.shape[1]], x[2,:],
                    color='#5189BB', lw=2, ls="-")
    states_xd, = ax11.plot(t[0:x.shape[1]], xref[0,0:x.shape[1]],
                    color='#BB5651', lw=2, ls="--")
    states_yd, = ax11.plot(t[0:x.shape[1]], xref[1,0:x.shape[1]],
                    color='#69BB51', lw=2, ls="--")
    states_zd, = ax11.plot(t[0:x.shape[1]], xref[2,0:x.shape[1]],
                    color='#5189BB', lw=2, ls="--")

    ax11.set_ylabel(r"$[states]$", rotation='vertical')
    ax11.set_xlabel(r"$[t]$", labelpad=5)
    ax11.legend([states_x, states_y, states_z, states_xd, states_yd, states_zd],
            [r'$x$', r'$y$', r'$z$', r'$x_d$', r'$y_d$', r'$z_d$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    #fig1.savefig("states_xyz.eps")
    fig1.savefig("states_xyz.png")
    fig1


    fig2, ax12 = fancy_plots_1()
    states_phi, = ax12.plot(t[0:x.shape[1]], x[3,:],
                    color='#BB5651', lw=2, ls="-")
    states_theta, = ax12.plot(t[0:x.shape[1]], x[4,:],
                    color='#69BB51', lw=2, ls="-")
    states_psi, = ax12.plot(t[0:x.shape[1]], x[5,:],
                    color='#5189BB', lw=2, ls="-")

    ax12.set_ylabel(r"$[states]$", rotation='vertical')
    ax12.set_xlabel(r"$[t]$", labelpad=5)
    ax12.legend([states_phi, states_theta, states_psi],
            [r'$\phi$', r'$\theta$', r'$\psi$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax12.grid(color='#949494', linestyle='-.', linewidth=0.5)

    #fig2.savefig("states_angles.eps")
    fig2.savefig("states_angles.png")
    fig2


    fig3, ax13 = fancy_plots_1()
    ## Axis definition necesary to fancy plots
    ax13.set_xlim((t[0], t[-1]))

    time_1, = ax13.plot(t[0:delta_t.shape[1]],delta_t[0,:],
                    color='#00429d', lw=2, ls="-")
    tsam1, = ax13.plot(t[0:t_sample.shape[1]],t_sample[0,:],
                    color='#9e4941', lw=2, ls="-.")

    ax13.set_ylabel(r"$[s]$", rotation='vertical')
    ax13.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
    ax13.legend([time_1,tsam1],
            [r'$t_{compute}$',r'$t_{sample}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax13.grid(color='#949494', linestyle='-.', linewidth=0.5)

    #fig3.savefig("time.eps")
    fig3.savefig("time.png")
    fig3

    fig4, ax14 = fancy_plots_1()
    errors_x, = ax14.plot(t[0:Ex.shape[1]], Ex[0,:],
                    color='#BB5651', lw=2, ls="-")
    errors_y, = ax14.plot(t[0:Ex.shape[1]], Ey[0,:],
                    color='#69BB51', lw=2, ls="-")
    errors_z, = ax14.plot(t[0:Ex.shape[1]], Ez[0,:],
                    color='#5189BB', lw=2, ls="-")

    ax14.set_ylabel(r"$[Errors]$", rotation='vertical')
    ax14.set_xlabel(r"$[t]$", labelpad=5)
    ax14.legend([errors_x, errors_y, errors_z],
            [r'$\tilde{h}_x$', r'$\tilde{h}_y$', r'$\tilde{h}_z$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax14.grid(color='#949494', linestyle='-.', linewidth=0.5)

    #fig4.savefig("errors.eps")
    fig4.savefig("errors_pos.png")
    fig4
  

    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')



if __name__ == '__main__':
    try:
        # Node Initialization
        rospy.init_node("Acados_controller",disable_signals=True, anonymous=True)

        # SUCRIBER
        velocity_subscriber = rospy.Subscriber("/dji_sdk/odometry", Odometry, odometry_call_back)
        
        # PUBLISHER
        velocity_message = TwistStamped()
        velocity_publisher = rospy.Publisher("/m100/velocityControl", TwistStamped, queue_size=10)

        odometry_sim_msg = Odometry()
        odom_sim_pub = rospy.Publisher('/dji_sdk/odometry', Odometry, queue_size=10)
    

        main(velocity_publisher, velocity_message, odom_sim_pub, odometry_sim_msg)
    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("\nError System")
        send_velocity_control([0, 0, 0, 0], velocity_publisher, velocity_message)
        pass
    else:
        print("Complete Execution")
        pass
