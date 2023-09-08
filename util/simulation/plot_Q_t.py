import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.unified0D_plus.apply_unified0D_plus  import *
from util.unified0D_plus.graph_to_junction_dict import *

from util.regression.neural_network.training_util import *
from util.tools.graph_handling import *
from util.tools.basic import *
import tensorflow as tf
import dgl
from dgl.data import DGLDataset
plt.rcParams.update({'font.size': 25})

num_time_steps = 1000
t = np.linspace(start = 0, stop = 2*np.pi, num = num_time_steps)
flow_amp = 60
q = -1*(flow_amp/2) * (np.cos(t)-1)
for i in range(t.size):
    t[i] =  i*0.2/t.size

plt.plot(t, q, "--", linewidth=4, color = "seagreen", label = "Unified0D+")

plt.xlabel("time (s)")
plt.ylabel("Q ($\mathrm{cm^3/s}$)")
plt.savefig(f"results/q_t_profile.pdf", bbox_inches='tight', format = "pdf")
