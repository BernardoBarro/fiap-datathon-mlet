from pathlib import Path

import mlflow
import numpy as np
import pandas as pd


EXPERIMENT_NAME = "fiap-datathon-bandit"
FINAL_SEED = 42
ROBUSTNESS_SEEDS = range(30)
HISTORY_RATIO = 0.80
PRIOR_ALPHA = 1
PRIOR_BETA = 1


def get_context(previous: int) -> str:
    if previous == 0:
        return "none"

    if previous == 1:
        return "one"

    return "two_plus"


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path)

    df = (
        df
        .sort_values("interaction_id")
        .reset_index(drop=True)
    )

    df["previous_group"] = df["previous"].apply(get_context)

    split_index = int(len(df) * HISTORY_RATIO)

    history_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return history_df, test_df


def evaluate_baseline(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    arm_performance = history_df.groupby("arm")["reward"].mean()
    selected_arm = arm_performance.idxmax()

    matches = test_df[
        test_df["arm"] == selected_arm
    ].copy()

    return {
        "selected_arm": selected_arm,
        "conversion_rate": float(matches["reward"].mean()),
        "coverage": float(len(matches) / len(test_df)),
        "evaluated_interactions": int(len(matches)),
        "conversions": int(matches["reward"].sum()),
    }


def initialize_posteriors(
    history_df: pd.DataFrame,
) -> tuple[list[str], dict, dict]:
    arms = sorted(history_df["arm"].dropna().unique())

    alpha = {}
    beta = {}

    for context in ("none", "one", "two_plus"):
        for arm in arms:
            alpha[(context, arm)] = PRIOR_ALPHA
            beta[(context, arm)] = PRIOR_BETA

    for context, arm, reward in history_df[
        ["previous_group", "arm", "reward"]
    ].itertuples(index=False, name=None):
        key = (context, arm)

        alpha[key] += int(reward)
        beta[key] += 1 - int(reward)

    return arms, alpha, beta


def run_thompson_replay(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> dict:
    arms, alpha, beta = initialize_posteriors(history_df)
    rng = np.random.default_rng(seed)

    evaluated_interactions = 0
    conversions = 0
    non_greedy_choices = 0

    for context, historical_arm, historical_reward in test_df[
        ["previous_group", "arm", "reward"]
    ].itertuples(index=False, name=None):

        posterior_means = {
            arm: alpha[(context, arm)]
            / (alpha[(context, arm)] + beta[(context, arm)])
            for arm in arms
        }

        samples = {
            arm: rng.beta(
                alpha[(context, arm)],
                beta[(context, arm)],
            )
            for arm in arms
        }

        chosen_arm = max(samples, key=samples.get)
        greedy_arm = max(posterior_means, key=posterior_means.get)

        if chosen_arm != greedy_arm:
            non_greedy_choices += 1

        if chosen_arm != historical_arm:
            continue

        reward = int(historical_reward)

        evaluated_interactions += 1
        conversions += reward

        key = (context, chosen_arm)
        alpha[key] += reward
        beta[key] += 1 - reward

    conversion_rate = (
        conversions / evaluated_interactions
        if evaluated_interactions > 0
        else 0.0
    )

    return {
        "conversion_rate": float(conversion_rate),
        "coverage": float(evaluated_interactions / len(test_df)),
        "evaluated_interactions": int(evaluated_interactions),
        "conversions": int(conversions),
        "non_greedy_rate": float(non_greedy_choices / len(test_df)),
    }


def calculate_robustness(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    runs = [
        run_thompson_replay(
            history_df,
            test_df,
            seed=seed,
        )
        for seed in ROBUSTNESS_SEEDS
    ]

    conversion_rates = [
        run["conversion_rate"]
        for run in runs
    ]

    coverages = [
        run["coverage"]
        for run in runs
    ]

    return {
        "mean_conversion_rate": float(np.mean(conversion_rates)),
        "std_conversion_rate": float(
            np.std(conversion_rates, ddof=1)
        ),
        "mean_coverage": float(np.mean(coverages)),
        "number_of_seeds": len(runs),
    }


def configure_mlflow(project_root: Path) -> None:
    database_path = project_root / "mlflow.db"

    tracking_uri = (
        "sqlite:///"
        + str(database_path.resolve()).replace("\\", "/")
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_baseline_run(
    baseline: dict,
    dataset_size: int,
    history_size: int,
    test_size: int,
) -> str:
    with mlflow.start_run(run_name="baseline") as run:
        mlflow.log_params({
            "policy": "best_historical_arm",
            "selected_arm": baseline["selected_arm"],
            "history_ratio": HISTORY_RATIO,
            "test_ratio": round(1 - HISTORY_RATIO, 2),
            "dataset_size": dataset_size,
            "history_size": history_size,
            "test_size": test_size,
        })

        mlflow.log_metrics({
            "conversion_rate": baseline["conversion_rate"],
            "coverage": baseline["coverage"],
            "evaluated_interactions": baseline[
                "evaluated_interactions"
            ],
            "conversions": baseline["conversions"],
        })

        mlflow.set_tags({
            "stage": "7-mlops",
            "project": "fiap-datathon-mlet",
        })

        return run.info.run_id


def log_thompson_run(
    thompson: dict,
    robustness: dict,
    baseline: dict,
    dataset_size: int,
    history_size: int,
    test_size: int,
) -> str:
    absolute_gain_pp = (
        thompson["conversion_rate"]
        - baseline["conversion_rate"]
    ) * 100

    relative_gain_pct = (
        (
            thompson["conversion_rate"]
            / baseline["conversion_rate"]
        )
        - 1
    ) * 100

    with mlflow.start_run(run_name="thompson-sampling") as run:
        mlflow.log_params({
            "algorithm": "thompson_sampling",
            "context": "previous_group",
            "arms": "cellular,telephone",
            "prior_alpha": PRIOR_ALPHA,
            "prior_beta": PRIOR_BETA,
            "seed": FINAL_SEED,
            "history_ratio": HISTORY_RATIO,
            "test_ratio": round(1 - HISTORY_RATIO, 2),
            "robustness_seeds": robustness["number_of_seeds"],
            "dataset_size": dataset_size,
            "history_size": history_size,
            "test_size": test_size,
        })

        mlflow.log_metrics({
            "conversion_rate": thompson["conversion_rate"],
            "coverage": thompson["coverage"],
            "evaluated_interactions": thompson[
                "evaluated_interactions"
            ],
            "conversions": thompson["conversions"],
            "non_greedy_rate": thompson["non_greedy_rate"],
            "absolute_gain_pp": absolute_gain_pp,
            "relative_gain_pct": relative_gain_pct,
            "robustness_mean_conversion_rate": robustness[
                "mean_conversion_rate"
            ],
            "robustness_std_conversion_rate": robustness[
                "std_conversion_rate"
            ],
            "robustness_mean_coverage": robustness[
                "mean_coverage"
            ],
        })

        mlflow.set_tags({
            "stage": "7-mlops",
            "project": "fiap-datathon-mlet",
            "evaluation": "offline_replay",
        })

        summary = {
            "baseline": baseline,
            "thompson_sampling": thompson,
            "robustness": robustness,
            "absolute_gain_pp": absolute_gain_pp,
            "relative_gain_pct": relative_gain_pct,
        }

        mlflow.log_dict(
            summary,
            "metrics_summary.json",
        )

        return run.info.run_id


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    data_path = (
        project_root
        / "data"
        / "processed"
        / "bank_marketing_processed.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset processado não encontrado em: {data_path}"
        )

    history_df, test_df = load_data(data_path)

    baseline = evaluate_baseline(
        history_df,
        test_df,
    )

    thompson = run_thompson_replay(
        history_df,
        test_df,
        seed=FINAL_SEED,
    )

    robustness = calculate_robustness(
        history_df,
        test_df,
    )

    configure_mlflow(project_root)

    baseline_run_id = log_baseline_run(
        baseline=baseline,
        dataset_size=len(history_df) + len(test_df),
        history_size=len(history_df),
        test_size=len(test_df),
    )

    thompson_run_id = log_thompson_run(
        thompson=thompson,
        robustness=robustness,
        baseline=baseline,
        dataset_size=len(history_df) + len(test_df),
        history_size=len(history_df),
        test_size=len(test_df),
    )

    absolute_gain_pp = (
        thompson["conversion_rate"]
        - baseline["conversion_rate"]
    ) * 100

    relative_gain_pct = (
        (
            thompson["conversion_rate"]
            / baseline["conversion_rate"]
        )
        - 1
    ) * 100

    print("\nMLflow tracking concluído.")
    print(f"Experimento: {EXPERIMENT_NAME}")
    print(f"Baseline run ID: {baseline_run_id}")
    print(f"Thompson run ID: {thompson_run_id}")

    print("\nBaseline")
    print(
        f"Conversão: "
        f"{baseline['conversion_rate'] * 100:.2f}%"
    )
    print(
        f"Cobertura: "
        f"{baseline['coverage'] * 100:.2f}%"
    )

    print("\nThompson Sampling")
    print(
        f"Conversão: "
        f"{thompson['conversion_rate'] * 100:.2f}%"
    )
    print(
        f"Cobertura: "
        f"{thompson['coverage'] * 100:.2f}%"
    )
    print(
        f"Ganho absoluto: "
        f"{absolute_gain_pp:.2f} p.p."
    )
    print(
        f"Ganho relativo: "
        f"{relative_gain_pct:.2f}%"
    )

    print("\nRobustez")
    print(
        f"Conversão média: "
        f"{robustness['mean_conversion_rate'] * 100:.2f}%"
    )
    print(
        f"Desvio padrão: "
        f"{robustness['std_conversion_rate'] * 100:.2f} p.p."
    )
    print(
        f"Cobertura média: "
        f"{robustness['mean_coverage'] * 100:.2f}%"
    )

    print(
        "\nPara visualizar os runs, execute na raiz do projeto:"
    )
    print(
        "mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000"
    )
    print("Depois acesse: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
