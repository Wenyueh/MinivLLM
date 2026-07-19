from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64  # Maximum number of tokens to generate (completion tokens only)
    ignore_eos: bool = False
    max_model_length: int | None = None  # Maximum total sequence length (prompt + completion)

    def __post_init__(self):
        if self.temperature <= 1e-10:
            raise ValueError("temperature must be greater than 0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if self.max_model_length is not None and self.max_model_length <= 0:
            raise ValueError("max_model_length must be greater than 0")
