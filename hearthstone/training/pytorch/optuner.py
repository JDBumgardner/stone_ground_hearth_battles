import optuna

from hearthstone.training.pytorch.ppo import ppo


def objective(trial:optuna.Trial):
    hparams = {
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
         "batch_size": trial.suggest_int("batch_size", 1, 10000, log=True),
         "num_workers": trial.suggest_int("num_workers", 1, 200, log=True),
         "ppo_epochs": trial.suggest_int("ppo_epochs", 1, 50),
         "ppo_epsilon": trial.suggest_float("ppo_epsilon", 0.01, 0.5, log=True),
         "policy_weight": trial.suggest_float("policy_weight", 0.1, 10, log=True),
         "entropy_weight": trial.suggest_float("entropy_weight", 1e-7, 1e-2, log = True),
         "nn_hidden_layers": trial.suggest_int("nn_hidden_layers", 0, 3),
         "nn_hidden_size": trial.suggest_int("nn_hidden_size", 10, 4096),
         "nn_shared": trial.suggest_categorical("nn_shared", [True, False]),
         "nn_activation": trial.suggest_categorical("nn_activation", ["relu", "gelu"])
    }
    if hparams["optimizer"] == "adam":
        hparams["adam_lr"] = trial.suggest_float("adam_lr", 1e-6, 1e-3, log=True)
    elif hparams["optimizer"] == "sgd":
        hparams["sgd_lr"] = trial.suggest_float("sgd_lr", 1e-6, 1e-3, log=True)
        hparams["sgd_momentum"] = trial.suggest_float("sgd_momentum", 0, 1)

    return ppo(hparams, 600, trial)


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_jobs=10)
    print(study.best_params)


if __name__ == '__main__':
    main()
