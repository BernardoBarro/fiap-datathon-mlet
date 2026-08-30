from pathlib import Path

import numpy as np
import pandas as pd


class RecommendationPolicy:
    CONTEXTS = ("none", "one", "two_plus")

    def __init__(
        self,
        data_path: Path | None = None,
        history_ratio: float = 0.80,
        seed: int = 42,
    ):
        project_root = Path(__file__).resolve().parents[1]

        self.data_path = data_path or (
            project_root
            / "data"
            / "processed"
            / "bank_marketing_processed.csv"
        )

        self.history_ratio = history_ratio
        self.rng = np.random.default_rng(seed)

        self.arms: list[str] = []
        self.alpha: dict[tuple[str, str], int] = {}
        self.beta: dict[tuple[str, str], int] = {}

        self._fit()

    @staticmethod
    def get_context(previous: int) -> str:
        if previous == 0:
            return "none"

        if previous == 1:
            return "one"

        return "two_plus"

    def _fit(self) -> None:
        df = pd.read_csv(self.data_path)

        df = (
            df
            .sort_values("interaction_id")
            .reset_index(drop=True)
        )

        split_index = int(len(df) * self.history_ratio)
        history_df = df.iloc[:split_index].copy()

        history_df["previous_group"] = history_df["previous"].apply(
            self.get_context
        )

        self.arms = sorted(history_df["arm"].dropna().unique())

        for context in self.CONTEXTS:
            for arm in self.arms:
                self.alpha[(context, arm)] = 1
                self.beta[(context, arm)] = 1

        for context, arm, reward in history_df[
            ["previous_group", "arm", "reward"]
        ].itertuples(index=False, name=None):
            key = (context, arm)

            self.alpha[key] += int(reward)
            self.beta[key] += 1 - int(reward)

    def posterior_estimates(self, context: str) -> dict[str, float]:
        return {
            arm: (
                self.alpha[(context, arm)]
                / (
                    self.alpha[(context, arm)]
                    + self.beta[(context, arm)]
                )
            )
            for arm in self.arms
        }

    def recommend(
        self,
        previous: int,
        mode: str = "deterministic",
    ) -> dict:
        context = self.get_context(previous)
        posterior = self.posterior_estimates(context)

        if mode == "thompson":
            scores = {
                arm: float(
                    self.rng.beta(
                        self.alpha[(context, arm)],
                        self.beta[(context, arm)],
                    )
                )
                for arm in self.arms
            }
        else:
            scores = posterior

        recommended_channel = max(
            scores,
            key=scores.get,
        )

        return {
            "context": context,
            "recommended_channel": recommended_channel,
            "mode": mode,
            "posterior_estimates": {
                arm: round(value, 6)
                for arm, value in posterior.items()
            },
            "decision_scores": {
                arm: round(float(value), 6)
                for arm, value in scores.items()
            },
        }
