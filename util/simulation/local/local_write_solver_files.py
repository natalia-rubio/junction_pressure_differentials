import numpy as np

def write_svpre_steady(anatomy, geo, flow_index, flow_params, cap_numbers, inlet_cap_number, num_time_steps):
    res_caps = cap_numbers
    res_caps.remove(inlet_cap_number)
    flow_name = f"flow_{flow_index}"
    svpre = f"mesh_and_adjncy_vtu data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-complete.mesh.vtu\n\
set_surface_id_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-complete.exterior.vtp 1\n\
set_surface_id_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp 2\n\
set_surface_id_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[0]}.vtp 3\n\
set_surface_id_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[1]}.vtp 4\n\
fluid_density 1.06\n\
fluid_viscosity 0.04\n\
initial_pressure 0\n\
initial_velocity 0.0001 {flow_params['vel_in']} 0.0001\n\
prescribed_velocities_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp\n\
bct_analytical_shape parabolic\n\
bct_period {num_time_steps*0.001}\n\
bct_point_number {num_time_steps}\n\
bct_fourier_mode_number 10\n\
bct_create data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_index}.flow\n\
bct_write_dat data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/bct.dat\n\
bct_write_vtp data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/bct.vtp\n\
pressure_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[0]}.vtp 0\n\
pressure_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[1]}.vtp 0\n\
noslip_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/walls_combined.vtp\n\
write_geombc data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/geombc.dat.1\n\
write_restart data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/restart.0.1"

    f = open(f"data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_name}_job.svpre", "w")
    f.write(svpre)
    f.close()
    return

def write_svpre_unsteady(anatomy, geo, flow_params, cap_numbers, inlet_cap_number, num_time_steps):
    res_caps = cap_numbers
    res_caps.remove(inlet_cap_number)
    flow_name = f"unsteady"
    flow_index = "unsteady"
    svpre = f"mesh_and_adjncy_vtu data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-complete.mesh.vtu\n\
set_surface_id_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-complete.exterior.vtp 1\n\
set_surface_id_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp 2\n\
set_surface_id_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[0]}.vtp 3\n\
set_surface_id_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[1]}.vtp 4\n\
fluid_density 1.06\n\
fluid_viscosity 0.04\n\
initial_pressure 0\n\
initial_velocity 0.0001 0.0001 0.0001\n\
prescribed_velocities_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp\n\
bct_analytical_shape parabolic\n\
bct_period {num_time_steps*0.002}\n\
bct_point_number {num_time_steps}\n\
bct_fourier_mode_number 10\n\
bct_create data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_index}.flow\n\
bct_write_dat data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/bct.dat\n\
bct_write_vtp data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/bct.vtp\n\
pressure_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[0]}.vtp 0\n\
pressure_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[1]}.vtp 0\n\
noslip_vtp data/synthetic_junctions/{anatomy}/{geo}/mesh-complete/walls_combined.vtp\n\
write_geombc data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/geombc.dat.1\n\
write_restart data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/restart.0.1"

    f = open(f"data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_name}_job.svpre", "w")

    f.write(svpre)
    f.close()
    return

def write_steady_inp(anatomy, geo, flow_index, flow_params, cap_numbers, inlet_cap_number, num_time_steps):

    res_caps = cap_numbers
    res_caps.remove(inlet_cap_number)
    flow_name = f"flow_{flow_index}"
    inp = f"Density: 1.06\n\
Viscosity: 0.04\n\
\n\
Number of Timesteps: {num_time_steps}\n\
Time Step Size: 0.001\n\
\n\
Number of Timesteps between Restarts: 100\n\
Number of Force Surfaces: 1\n\
Surface ID's for Force Calculation: 1\n\
Force Calculation Method: Velocity Based\n\
Print Average Solution: True\n\
Print Error Indicators: False\n\
\n\
Time Varying Boundary Conditions From File: True\n\
\n\
Step Construction: 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1\n\
\n\
Number of Resistance Surfaces: 2\n\
List of Resistance Surfaces: {res_caps[0]} {res_caps[1]}\n\
Resistance Values: {flow_params['res_1']} {flow_params['res_2']}\n\
\n\
Pressure Coupling: Implicit\n\
Number of Coupled Surfaces: 2\n\
\n\
Backflow Stabilization Coefficient: 0.2\n\
Residual Control: True\n\
Residual Criteria: 0.00001\n\
Minimum Required Iterations: 1\n\
svLS Type: NS\n\
Number of Krylov Vectors per GMRES Sweep: 10\n\
Number of Solves per Left-hand-side Formation: 1\n\
Tolerance on Momentum Equations: 0.01\n\
Tolerance on Continuity Equations: 0.01\n\
Tolerance on svLS NS Solver: 0.01\n\
Maximum Number of Iterations for svLS NS Solver: 1\n\
Maximum Number of Iterations for svLS Momentum Loop: 2\n\
Maximum Number of Iterations for svLS Continuity Loop: 400\n\
Time Integration Rule: Second Order\n\
Time Integration Rho Infinity: 0.5\n\
Flow Advection Form: Convective\n\
Quadrature Rule on Interior: 2\n\
Quadrature Rule on Boundary: 3"

    f = open(f"data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_name}_solver.inp", "w")
    f.write(inp)
    f.close()
    return

def write_unsteady_inp(anatomy, geo, flow_params, cap_numbers, inlet_cap_number, num_time_steps):
    res_caps = cap_numbers
    res_caps.remove(inlet_cap_number)
    flow_name = f"unsteady"
    flow_index = "unsteady"
    inp = f"Density: 1.06\n\
Viscosity: 0.04\n\
\n\
Number of Timesteps: {num_time_steps}\n\
Time Step Size: 0.002\n\
\n\
Number of Timesteps between Restarts: 1\n\
Number of Force Surfaces: 1\n\
Surface ID's for Force Calculation: 1\n\
Force Calculation Method: Velocity Based\n\
Print Average Solution: True\n\
Print Error Indicators: False\n\
\n\
Time Varying Boundary Conditions From File: True\n\
\n\
Step Construction: 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1\n\
\n\
Number of Resistance Surfaces: 2\n\
List of Resistance Surfaces: {res_caps[0]} {res_caps[1]}\n\
Resistance Values: {flow_params['res_1']} {flow_params['res_2']}\n\
\n\
Pressure Coupling: Implicit\n\
Number of Coupled Surfaces: 2\n\
\n\
Backflow Stabilization Coefficient: 0.2\n\
Residual Control: True\n\
Residual Criteria: 0.00001\n\
Minimum Required Iterations: 1\n\
svLS Type: NS\n\
Number of Krylov Vectors per GMRES Sweep: 10\n\
Number of Solves per Left-hand-side Formation: 1\n\
Tolerance on Momentum Equations: 0.01\n\
Tolerance on Continuity Equations: 0.01\n\
Tolerance on svLS NS Solver: 0.01\n\
Maximum Number of Iterations for svLS NS Solver: 1\n\
Maximum Number of Iterations for svLS Momentum Loop: 2\n\
Maximum Number of Iterations for svLS Continuity Loop: 400\n\
Time Integration Rule: Second Order\n\
Time Integration Rho Infinity: 0.5\n\
Flow Advection Form: Convective\n\
Quadrature Rule on Interior: 2\n\
Quadrature Rule on Boundary: 3"

    f = open(f"data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_name}_solver.inp", "w")
    f.write(inp)
    f.close()
    return

def write_steady_flow(anatomy, geo, flow_index, flow_amp, cap_number, num_time_steps):
    flow_name = f"flow_{flow_index}"
    flow = ""
    t = np.linspace(start = 0, stop = num_time_steps, num = num_time_steps)
    q = t*0
    for i in range(t.size):
        if i < 200:
            q[i] = -1 * flow_amp * 0.5 * (1 - np.cos(np.pi * i / 200))
        else:
            q[i] = -1 * flow_amp

        flow = flow + "%1.3f %1.3f\n" %(i*0.001, q[i])
    f = open(f"data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_index}.flow", "w")
    f.write(flow)
    f.close()
    return

def write_unsteady_flow(anatomy, geo, flow_amp, cap_number, num_time_steps):
    flow_name = f"unsteady"
    flow_index = "unsteady"
    flow = ""
    t = np.linspace(start = 0, stop = 2*np.pi, num = num_time_steps)
    q = (flow_amp/2) * (np.cos(t)-1)
    for i in range(t.size):
        flow = flow + "%1.3f %1.3f\n" %(i*0.2/t.size, q[i])

    f = open(f"data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_index}.flow", "w")
    f.write(flow)
    f.close()
    return
