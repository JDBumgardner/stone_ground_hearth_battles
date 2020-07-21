import optuna


def main():
    study = optuna.create_study(storage="sqlite:///../../../data/learning/pytorch/optuna/study.db",
                                study_name="ppo_study", direction="maximize", load_if_exists=True)
    df = study.trials_dataframe()
    print(df)
    print(study.best_trial)


if __name__ == '__main__':
    main()