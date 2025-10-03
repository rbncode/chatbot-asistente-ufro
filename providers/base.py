from typing import Protocol, List, Dict, Any


class Provider(Protocol):
    @property
    def name(self) -> str:
        ...

    def chat(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> str:
        ...