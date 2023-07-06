import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *

def get_coefs_steady(anatomy, rm_low_r2 = True):

    num_outlets = 2
    char_val_dict = load_dict(f"/home/nrubio/Desktop/aorta_synthetic_data_dict_steady_{anatomy}")
    char_val_dict.update({"coef_a": [],
                    "coef_b": []})
    to_rm = []
    for outlet_ind, geo in enumerate(char_val_dict["name"]):

        flow_list = char_val_dict["flow_list"][outlet_ind]
        dP_list = char_val_dict["dP_list"][outlet_ind]

        Q = np.asarray(flow_list).reshape(-1,1)
        dP = np.asarray(dP_list).reshape(-1,1)
        X = np.hstack([np.square(Q), Q])

        coefs, residuals, t, q = np.linalg.lstsq(X, dP, rcond=None);
        if (np.linalg.norm(residuals)/(1333**2))> 0.01:
            print(f"geo: {geo} {(np.linalg.norm(residuals)/(1333**2))}")

        r2 = get_r2(X, dP, coefs.reshape(-1,1))
        a = coefs[0][0]; b = coefs[1][0]
        if get_r2(X, dP, coefs.reshape(-1,1)) < 0.8: #0.95:
            to_rm.append(outlet_ind)
            print(f"{geo} r2: {r2}")
        char_val_dict["coef_a"].append(a)
        char_val_dict["coef_b"].append(b)

    if rm_low_r2:
        print(f"Removing {len(to_rm)} outlets for low r2 values.")
        to_keep = []
        for i in range(int(len(char_val_dict["name"])/2)):
            if (2*i in to_rm) or 2*i+1 in to_rm:
                continue
            else:
                to_keep.append(2*i); to_keep.append(2*i + 1)

        for key in char_val_dict:
            try:
                char_val_dict[key] = [char_val_dict[key][ind] for ind in to_keep]
            except:
                continue

    save_dict(char_val_dict, f"data/characteristic_value_dictionaries/{anatomy}_synthetic_data_dict_steady")
    return

def remove_outlier_coefs(anatomy, sd_tol):
    char_val_dict = load_dict(f"/home/nrubio/Desktop/aorta_synthetic_data_dict_steady_{anatomy}")

    outlier_inds, non_outlier_inds = get_outlier_inds(char_val_dict["coef_a"][::2], m = sd_tol)

    non_outlier_inds = [2*ind for ind in non_outlier_inds] + [2*ind+1 for ind in non_outlier_inds]
    non_outlier_inds.sort()

    for key in char_val_dict.keys():
        try:
            full_array = char_val_dict[key]
            char_val_dict[key] = list(np.asarray(char_val_dict[key])[2*np.asarray(non_outlier_inds).astype(int)])
        except:
            continue

    print(f"a outlier_inds: {outlier_inds}")
    save_dict(char_val_dict, f"data/characteristic_value_dictionaries/{anatomy}_synthetic_data_dict_steady")
    return

def get_geo_scalings_steady(anatomy):

    plt.style.use('dark_background')

    char_val_dict = load_dict(f"data/characteristic_value_dictionaries/{anatomy}_synthetic_data_dict_steady")
    scaling_dict = {}
    to_normalize = ["outlet_radius","inlet_radius", "angle", "coef_a", "coef_b"]
    for value in to_normalize:
        
        scaling_dict.update({value: [np.mean(char_val_dict[value]), np.std(char_val_dict[value])]})

        plt.clf()
        plt.hist(char_val_dict[value], bins = 30, alpha = 0.5, label = "outlet1")
        plt.hist(char_val_dict[value], bins = 30, alpha = 0.5, label = "outlet2")
        plt.xlabel(value); plt.ylabel("frequency"); plt.title("Synthetic Aorta Data");
        plt.legend()
        plt.savefig(f"results/synthetic_data_trends/geo_dist/{value}_both.png", bbox_inches='tight', transparent=True, format = "png")

        plt.clf()
        plt.scatter(char_val_dict[value], char_val_dict["coef_a"])
        plt.xlabel(value); plt.ylabel("a"); plt.title("Synthetic Aorta Data");
        plt.savefig(f"results/synthetic_data_trends/a_trends/{value}.png", bbox_inches='tight', transparent=True, format = "png")

        plt.clf()
        plt.scatter(char_val_dict[value], np.asarray(char_val_dict["coef_b"]))
        plt.xlabel(value); plt.ylabel("b"); plt.title("Synthetic Aorta Data");
        plt.savefig(f"results/synthetic_data_trends/b_trends/{value}.png", bbox_inches='tight', transparent=True, format = "png")

    save_dict(scaling_dict, f"data/scaling_dictionaries/{anatomy}_scaling_dict_steady")
    return
