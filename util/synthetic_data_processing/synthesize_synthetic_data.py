import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *

def get_coefs(anatomy, rm_low_r2 = True, unsteady = False):

    num_outlets = 2
    char_val_dict = load_dict(f"data/characteristic_value_dictionaries/{anatomy}_synthetic_data_dict")
    char_val_dict.update({"coef_a": [],
                    "coef_b": [],
                    "coef_L": []})
    to_rm = []
    for outlet_ind, geo in enumerate(char_val_dict["name"]):

        flow_list = char_val_dict["flow_list"][outlet_ind]
        dP_list = char_val_dict["dP_list"][outlet_ind]

        Q = np.asarray(flow_list).reshape(-1,1)


        r2_unsteady = 0; r2_steady = 0
        if unsteady:
            Q_unsteady = char_val_dict["unsteady_flow_list"][outlet_ind].reshape(-1,1)
            dQdt = char_val_dict["unsteady_flow_der_list"][outlet_ind].reshape(-1,1)
            dP_unsteady = char_val_dict["unsteady_dP_list"][outlet_ind].reshape(-1,1)
            # dP_steady_pred = a*np.square(Q_unsteady) + b*Q_unsteady
            # dP_unsteady_comp = dP_unsteady - dP_steady_pred
            #
            # X_unsteady = dQdt.reshape(-1,1)
            # coefs, residuals, t, q = np.linalg.lstsq(X_unsteady, dP_unsteady_comp, rcond=None);
            # r2_unsteady = get_r2(X_unsteady, dP_unsteady_comp, coefs.reshape(-1,1))
            #
            # L = coefs[0]
            # char_val_dict["coef_L"].append(L)
            X_unsteady = np.hstack([np.square(Q_unsteady), Q_unsteady, dQdt])

            #import pdb; pdb.set_trace()
            coefs, residuals, t, q = np.linalg.lstsq(X_unsteady, dP_unsteady, rcond=None);

            r2_unsteady = get_r2(X_unsteady, dP_unsteady, coefs.reshape(-1,1))
            print(r2_unsteady)
            print(residuals)
            print(f"Error: {np.sqrt(residuals/dP_unsteady.size)/1333}")
            #print(coefs)
            a = coefs[0][0]; b = coefs[1][0]
            char_val_dict["coef_a"].append(a); char_val_dict["coef_b"].append(b)
            L = coefs[2][0]
            char_val_dict["coef_L"].append(L)
        else:
            dP = np.asarray(dP_list).reshape(-1,1)
            X = np.hstack([np.square(Q), Q])
            coefs, residuals, t, q = np.linalg.lstsq(X, dP, rcond=None);
            if (np.linalg.norm(residuals)/(1333**2))> 0.01:
                print(f"geo: {geo} {(np.linalg.norm(residuals)/(1333**2))}")

            a = coefs[0][0]; b = coefs[1][0]
            char_val_dict["coef_a"].append(a)
            char_val_dict["coef_b"].append(b)
            r2_steady = get_r2(X, dP, coefs.reshape(-1,1))


        if r2_steady < 0.90 and r2_unsteady < 0.90:
            to_rm.append(outlet_ind)
            print(f"{geo} Steady R2: {r2_steady}.  Unsteady R2: {r2_unsteady}.")

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

    save_dict(char_val_dict, f"data/characteristic_value_dictionaries/{anatomy}_synthetic_data_dict")
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

def get_geo_scalings(anatomy, unsteady = False):

    plt.style.use('dark_background')

    char_val_dict = load_dict(f"data/characteristic_value_dictionaries/{anatomy}_synthetic_data_dict")
    scaling_dict = {}
    to_normalize = ["outlet_radius","inlet_radius","outlet_area","inlet_area", "angle", "coef_a", "coef_b", "inlet_length", "outlet_length"]
    if unsteady:
        to_normalize.append("coef_L")
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
        plt.savefig(f"results/synthetic_data_trends/L_trends/{value}.png", bbox_inches='tight', transparent=True, format = "png")

        if unsteady:
            plt.clf()
            plt.scatter(char_val_dict[value], np.asarray(char_val_dict["coef_L"]))
            plt.xlabel(value); plt.ylabel("L"); plt.title("Synthetic Aorta Data");
            plt.savefig(f"results/synthetic_data_trends/L_trends/{value}.png", bbox_inches='tight', transparent=True, format = "png")
    if unsteady:
        save_dict(scaling_dict, f"data/scaling_dictionaries/{anatomy}_scaling_dict")
    else:
        save_dict(scaling_dict, f"data/scaling_dictionaries/{anatomy}_scaling_dict_steady")
    return
