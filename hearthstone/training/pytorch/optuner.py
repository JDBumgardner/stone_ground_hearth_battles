import joblib
import optuna

from hearthstone.training.pytorch.ppo import PPOHyperparameters, PPOLearner


def objective(trial: optuna.Trial):
    hparams = PPOHyperparameters({
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
        "batch_size": trial.suggest_int("batch_size", 1, 4096, log=True),
        "ppo_epochs": trial.suggest_int("ppo_epochs", 1, 40),
        "ppo_epsilon": trial.suggest_float("ppo_epsilon", 0.01, 0.5, log=True),
        "policy_weight": trial.suggest_float("policy_weight", 0.3, 3, log=True),
        "entropy_weight": trial.suggest_float("entropy_weight", 1e-7, 1e-2, log=True),
        "nn.hidden_layers": trial.suggest_int("nn.hidden_layers", 0, 3),
        "normalize_observations": trial.suggest_categorical("normalize_observations", [True, False]),
        "gradient_clipping": trial.suggest_float("gradient_clipping", 0.5, 0.5),
        "normalize_advantage": trial.suggest_categorical("normalize_advantage", [True, False]),
    })
    hparams["num_workers"] = trial.suggest_int("num_workers", 1, hparams["batch_size"], log=True)

    if hparams["optimizer"] == "adam":
        hparams["adam.lr"] = trial.suggest_float("adam.lr", 1e-6, 1e-3, log=True)
    elif hparams["optimizer"] == "sgd":
        hparams["sgd_lr"] = trial.suggest_float("sgd_lr", 1e-6, 1e-3, log=True)
        hparams["sgd_momentum"] = trial.suggest_float("sgd_momentum", 0.0, 1.0)

    if hparams["nn.hidden_layers"] > 0:
        hparams["nn.hidden_size"] = trial.suggest_int("nn.hidden_size", 32, 2048)
        hparams["nn.shared"] = trial.suggest_categorical("nn.shared", [True, False])
        hparams["nn.activation"] = trial.suggest_categorical("nn.activation", ["relu", "gelu", "tanh"])

    ppo_learner = PPOLearner(hparams, 600, trial)
    return ppo_learner.run()


def main():
    """
    A wise man once said, "You can optuna neural net but you can't optuna fish." - Albert Einstein.

    Returns: No returns. No refunds. No shirt. No service.
    """
    study = optuna.create_study(
        storage="postgres://localhost/optuna", study_name="ppo_study",
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.NopPruner())
    try:
        try:
            with joblib.parallel_backend("multiprocessing"):
                study.optimize(objective, n_jobs=10, catch=(RuntimeError,))
        except KeyboardInterrupt:
            pass
    except Exception as e:
        print(e)
    print(study.best_params)


if __name__ == '__main__':
    main()
