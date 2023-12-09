from knn import *
from dec_tree import *
from lin_reg import *
from svr import *
from gpr import *

import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.search.optuna import OptunaSearch
import os
import random
import pdb

ray.init(
    _system_config={
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": "/home/nrubio/Desktop"}},
        )
    },
    num_cpus=1,
    num_gpus=3,
)
anatomy = "Aorta_rand"; num_geos_steady = 110; num_geos_unsteady = 110


def objective_dt(config):
    print(config)
    #import pdb; pdb.set_trace()
    reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_dt_model_steady(anatomy, num_geos_steady, seed = 0, hyperparams = config)
    metric = dP_loss_val_steady
    train.report({"inference_performance": float(metric)})  # Report to Tune
    return

def objective_knn(config):
    print(config)
    #import pdb; pdb.set_trace()
    reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_knn_model_steady(anatomy, num_geos_steady, seed = 0, hyperparams = config)
    metric = dP_loss_val_steady
    train.report({"inference_performance": float(metric)})  # Report to Tune
    return

def objective_svr(config):
    print(config)
    #import pdb; pdb.set_trace()
    reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_svr_model_steady(anatomy, num_geos_steady, seed = 0, hyperparams = config)
    metric = dP_loss_val_steady
    train.report({"inference_performance": float(metric)})  # Report to Tune
    return

def objective_gpr(config):
    print(config)
    #import pdb; pdb.set_trace()
    reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_gpr_model_steady(anatomy, num_geos_steady, seed = 0, hyperparams = config)
    metric = dP_loss_val_steady
    train.report({"inference_performance": float(metric)})  # Report to Tune
    return

def main():

    algo = OptunaSearch()
    storage_path = os.path.expanduser(f"/home/nrubio/Desktop/junction_pressure_differentials/results/hyperparameter_optimization")

    # DECISION TREE
    # search_space = {
    # "max_depth": tune.randint(2, 10),
    # "min_samples_leaf": tune.randint(1,10)}
    # exp_name = "dt"
    # path = os.path.join(storage_path, exp_name)
    # objective_with_gpu = tune.with_resources(objective_dt, {"gpu": 3})
    # tuner = tune.Tuner(
    #     trainable = objective_with_gpu,
    #     tune_config=tune.TuneConfig(
    #         metric="inference_performance", mode="min", search_alg=algo,
    #         num_samples=100
    #         ),
    #     run_config=train.RunConfig(storage_path=storage_path, name=exp_name),
    #         param_space=search_space,)
    # results = tuner.fit()
    # print("Best config is:", results.get_best_result().config)

    # KNN
    # search_space = {
    # "n_neighbors": tune.randint(2, 30)}
    # exp_name = "knn"
    # path = os.path.join(storage_path, exp_name)
    #
    # objective_with_gpu = tune.with_resources(objective_knn, {"gpu": 3})
    # tuner = tune.Tuner(
    #     trainable = objective_with_gpu,
    #     tune_config=tune.TuneConfig(
    #         metric="inference_performance", mode="min", search_alg=algo,
    #         num_samples=40
    #         ),
    #     run_config=train.RunConfig(storage_path=storage_path, name=exp_name),
    #         param_space=search_space,)
    # results = tuner.fit()
    # print("Best config is:", results.get_best_result().config)

    # SVR
    # search_space = {"C": tune.loguniform(1e-2, 1e1),
    # "epsilon": tune.loguniform(1e-2, 1e1)}
    # exp_name = "svr"
    # path = os.path.join(storage_path, exp_name)
    #
    # objective_with_gpu = tune.with_resources(objective_svr, {"gpu": 3})
    # tuner = tune.Tuner(
    #     trainable = objective_with_gpu,
    #     tune_config=tune.TuneConfig(
    #         metric="inference_performance", mode="min", search_alg=algo,
    #         num_samples=100
    #         ),
    #     run_config=train.RunConfig(storage_path=storage_path, name=exp_name),
    #         param_space=search_space,)
    # results = tuner.fit()
    # print("Best config is:", results.get_best_result().config)


    # GPR
    search_space = {"alpha": tune.loguniform(1e-5, 1e1),
    "length_scale": tune.loguniform(1e-2, 1e1)}
    exp_name = "gpr"
    path = os.path.join(storage_path, exp_name)

    objective_with_gpu = tune.with_resources(objective_gpr, {"gpu": 3})
    tuner = tune.Tuner(
        trainable = objective_with_gpu,
        tune_config=tune.TuneConfig(
            metric="inference_performance", mode="min", search_alg=algo,
            num_samples=100
            ),
        run_config=train.RunConfig(storage_path=storage_path, name=exp_name),
            param_space=search_space,)
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

if __name__ == "__main__":
    main()


# print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('Model','Steady (Train)','Steady (Val)','Unsteady (Train)', 'Unsteady (Val)'))
#
# reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_knn_model_steady(anatomy, num_geos_steady, seed = 0)
# reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_knn_model_unsteady(anatomy, num_geos_unsteady, seed = 0)
# print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('KNN',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))
#
#
# reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_dt_model_steady(anatomy, num_geos_steady, seed = 0)
# reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_dt_model_unsteady(anatomy, num_geos_unsteady, seed = 0)
# print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('DT',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))
#
# reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_lin_reg_model_steady(anatomy, num_geos_steady, seed = 0)
# reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_lin_reg_model_unsteady(anatomy, num_geos_unsteady, seed = 0)
# print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('LR',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))
#
# reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_svr_model_steady(anatomy, num_geos_steady, seed = 0)
# reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_svr_model_unsteady(anatomy, num_geos_unsteady, seed = 0)
# print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('SVR',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))
#
# reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_gpr_model_steady(anatomy, num_geos_steady, seed = 0)
# reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_gpr_model_unsteady(anatomy, num_geos_unsteady, seed = 0)
# print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('GPR',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))
