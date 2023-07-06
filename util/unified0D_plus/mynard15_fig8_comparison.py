from unified0D_plus import *
import matplotlib.pyplot as plt
import sys
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rc('text', usetex=True)
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
npts = 100

"""
Set base case to imitate Mynard 2015, Figure 8.

From figure:
Inlet Re: 1817.  Flow ratio: 0.5.  Outlet Angles: pi/3.

"""
re = 1817
viscosity = 0.04
density = 1.06
r_in = 0.55
r_out = r_in

U_in = re * viscosity / (density * 2 * r_in)
U_out = (U_in * (np.pi * r_in**2) / 2)
U = np.asarray([U_in, -1*U_out, -1*U_out])
A = np.asarray([np.pi*r_in**2, np.pi*r_out**2, np.pi*r_out**2])
theta = np.asarray([np.pi, np.pi/4, -np.pi/4])

"""
Mynard Paper Figure 8A
Coefficient K vs. Inlet Re
"""
re_dig = [455.88235294117646, 676.4705882352941, 897.0588235294117, 1126.4705882352941, 1352.9411764705883, 1582.3529411764707, 1797.058823529412]
K_dig = [0.3468354430379747, 0.3468354430379747, 0.3468354430379747, 0.3468354430379747, 0.3468354430379747, 0.3367088607594937, 0.3417721518987342]

re_arr  = np.linspace(re_dig[0], re_dig[-1], npts)
K_arr = re_arr * 0

for i in range(npts):
    U_in = re_arr[i] * viscosity / (density * 2 * r_in)
    U_out = (U_in * (np.pi * r_in**2) / 2) / (np.pi*r_out**2)
    U = np.asarray([U_in, -1*U_out, -1*U_out])
    C, K = junction_loss_coeff(U, A, theta)
    K_arr[i] = K[0]

plt.clf()
plt.plot(re_arr, K_arr, label = "Re-Implementation")
plt.scatter(re_dig, K_dig, label = "Digitized")
plt.xlabel("Re"); plt.ylabel("Loss Coefficient $K$"); plt.legend(); plt.ylim([0,1])
plt.savefig(f"results/unified0D_plus/mynard_fig_8a_comp.pdf", bbox_inches='tight', transparent=True, format = "pdf")

"""
Mynard Paper Figure 8B
Coefficient K vs. Flow Fraction
"""
flow_frac_dig = [0.09902912621359224, 0.14692556634304207, 0.19741100323624597, 0.2504854368932039, 0.3048543689320389, 0.36310679611650487, 0.42394822006472493, 0.4886731391585761, 0.5443365695792881, 0.6090614886731391, 0.6750809061488674, 0.7527508090614887, 0.822653721682848, 0.8640776699029127, 0.8977346278317153]
K_dig = [0.8129032258064517, 0.7290322580645161, 0.6580645161290323, 0.5806451612903225, 0.5182795698924731, 0.44946236559139785, 0.3913978494623656, 0.34408602150537637, 0.31827956989247314, 0.2860215053763441, 0.2623655913978495, 0.25591397849462366, 0.2623655913978495, 0.2752688172043011, 0.2903225806451613]

flow_frac_arr  = np.linspace(flow_frac_dig[0], flow_frac_dig[-1], npts)
K_arr = flow_frac_arr * 0

for i in range(npts):

    U_out1 = flow_frac_arr[i] * (U_in * np.pi * r_in**2)  / (np.pi*r_out**2)
    U_out2 = (1-flow_frac_arr[i]) * (U_in * np.pi * r_in**2)  / (np.pi*r_out**2)

    U = np.asarray([U_in, -1*U_out1, -1*U_out2])
    C, K = junction_loss_coeff(U, A, theta)
    K_arr[i] = K[0]

plt.clf()
plt.plot(flow_frac_arr, K_arr, label = "Re-Implementation")
plt.scatter(flow_frac_dig, K_dig, label = "Digitized")
plt.xlabel("Flow Fraction"); plt.ylabel("Loss Coefficient $K$"); plt.legend(); plt.ylim([0,1])
plt.savefig(f"results/unified0D_plus/mynard_fig_8b_comp.pdf", bbox_inches='tight', transparent=True, format = "pdf")


"""
Mynard Paper Figure 8C
Coefficient K vs. Branch Angle
"""
branch_angle_dig = [14.623655913978494, 22.457757296466973, 29.216589861751153, 36.74347158218126, 44.57757296466974, 51.643625192012294, 57.94162826420891, 64.39324116743472, 70.38402457757297, 75.76036866359448, 81.29032258064517, 88.35637480798772, 93.57910906298004, 98.80184331797236, 104.02457757296467, 109.0937019969278, 114.16282642089094, 119.69278033794164]
K_dig = [0.1723404255319149, 0.2, 0.23191489361702128, 0.2765957446808511, 0.33617021276595743, 0.39148936170212767, 0.4531914893617021, 0.5212765957446809, 0.5872340425531914, 0.6531914893617021, 0.7212765957446808, 0.8106382978723404, 0.8808510638297873, 0.9553191489361702, 1.025531914893617, 1.1, 1.1702127659574468, 1.2468085106382978]

U_out = (U_in * (np.pi * r_in**2) / 2)
U = np.asarray([U_in, -1*U_out, -1*U_out])
A = np.asarray([np.pi*r_in**2, np.pi*r_out**2, np.pi*r_out**2])

branch_angle_arr  = np.linspace(branch_angle_dig[0], branch_angle_dig[-1], npts)
K_arr = branch_angle_arr * 0

for i in range(npts):
    theta = np.asarray([np.pi, np.pi*branch_angle_arr[i]/180, -np.pi*branch_angle_arr[i]/180])
    C, K = junction_loss_coeff(U, A, theta)
    K_arr[i] = K[0]

plt.clf()
plt.plot(branch_angle_arr, K_arr, label = "Re-Implementation")
plt.scatter(branch_angle_dig, K_dig, label = "Digitized")
plt.xlabel("Branch Angle"); plt.ylabel("Loss Coefficient $K$"); plt.legend();
plt.savefig(f"results/unified0D_plus/mynard_fig_8c_comp.pdf", bbox_inches='tight', transparent=True, format = "pdf")
