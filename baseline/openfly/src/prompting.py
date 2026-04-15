from typing import Optional


SYS_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)


def _format_system_prompt(system_prompt: str) -> str:
    return f"<<SYS>\n{system_prompt.strip()}\n<</SYS>>\n\n"


class LLaMa2ChatPromptBuilder:
    """Minimal OpenFly-compatible prompt builder."""

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        self.system_prompt = _format_system_prompt(SYS_PROMPT if system_prompt is None else system_prompt)
        self.prompt = ""
        self.turn_count = 0

    def add_turn(self, role: str, message: str) -> str:
        expected_role = "human" if (self.turn_count % 2 == 0) else "gpt"
        if role != expected_role:
            raise ValueError(f"Unexpected role={role!r} at turn={self.turn_count}, expected={expected_role!r}")

        message = message.replace("<image>", "").strip()
        if self.turn_count == 0:
            wrapped = f"[INST] {self.system_prompt}{message} [/INST] "
        elif role == "human":
            wrapped = f"[INST] {message} [/INST] "
        else:
            wrapped = f"{message if message != '' else ' '}</s>"

        self.prompt += wrapped
        self.turn_count += 1
        return wrapped

    def get_prompt(self) -> str:
        return self.prompt.removeprefix("<s>").rstrip()


def normalize_instruction(text: str) -> str:
    return " ".join(text.replace("\r\n", " ").replace("\n", " ").strip().split()).lower()


def build_openfly_prompt(instruction: str) -> str:
    prompt_builder = LLaMa2ChatPromptBuilder()
    prompt_builder.add_turn("human", f"What action should the robot take to {normalize_instruction(instruction)}?")
    return prompt_builder.get_prompt()


def build_openfly_prompt_with_answer(instruction: str, answer: str) -> str:
    prompt_builder = LLaMa2ChatPromptBuilder()
    prompt_builder.add_turn("human", f"What action should the robot take to {normalize_instruction(instruction)}?")
    prompt_builder.add_turn("gpt", answer)
    return prompt_builder.get_prompt()
