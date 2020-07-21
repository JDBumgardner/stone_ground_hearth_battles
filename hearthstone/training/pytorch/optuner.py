import joblib
import optuna
from optuna.pruners import HyperbandPruner

from hearthstone.training.pytorch.ppo import ppo


def objective(trial: optuna.Trial):
    hparams = {
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
         "batch_size": trial.suggest_int("batch_size", 1, 4096, log=True),
         "num_workers": trial.suggest_int("num_workers", 1, 200, log=True),
         "ppo_epochs": trial.suggest_int("ppo_epochs", 1, 40),
         "ppo_epsilon": trial.suggest_float("ppo_epsilon", 0.01, 0.5, log=True),
         "policy_weight": trial.suggest_float("policy_weight", 0.3, 3, log=True),
         "entropy_weight": trial.suggest_float("entropy_weight", 1e-7, 1e-2, log=True),
         "nn_hidden_layers": trial.suggest_int("nn_hidden_layers", 0, 3),
         "normalize_observations": trial.suggest_categorical("normalize_observations", [True, False]),
    }
    if hparams["optimizer"] == "adam":
        hparams["adam_lr"] = trial.suggest_float("adam_lr", 1e-6, 1e-3, log=True)
    elif hparams["optimizer"] == "sgd":
        hparams["sgd_lr"] = trial.suggest_float("sgd_lr", 1e-6, 1e-3, log=True)
        hparams["sgd_momentum"] = trial.suggest_float("sgd_momentum", 0.0, 1.0)

    if hparams["nn_hidden_layers"] > 0:
        hparams["nn_hidden_size"] = trial.suggest_int("nn_hidden_size", 32, 2048)
        hparams["nn_shared"] = trial.suggest_categorical("nn_shared", [True, False])
        hparams["nn_activation"] = trial.suggest_categorical("nn_activation", ["relu", "gelu"])
    return ppo(hparams, 600, trial)


def main():
    study = optuna.create_study(
        storage="sqlite:///../../../data/learning/pytorch/optuna/study.db", study_name="ppo_study",
                                direction="maximize",
                                load_if_exists=True,
                                pruner=optuna.pruners.HyperbandPruner())

    try:
        try:
            with joblib.parallel_backend("multiprocessing"):
                study.optimize(objective, n_jobs=10, timeout=700)
        except KeyboardInterrupt:
            pass
    except e:
        print(e)
    print(study.best_params)


if __name__ == '__main__':
    main()
