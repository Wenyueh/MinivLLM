from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64  # Maximum number of tokens to generate (completion tokens only)
    ignore_eos: bool = False
    max_model_length: int | None = None  # Maximum total sequence length (prompt + completion)

    def __post_init__(self):
        assert self.temperature >= 0, "greedy sampling is not permitted"