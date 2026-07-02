from abc import ABC, abstractmethod
from typing import AsyncGenerator

class BaseProvider(ABC):
    @abstractmethod
    async def stream_response(self, messages: list) -> AsyncGenerator[str, None]:
        """
        Accepts a list of chat history messages (e.g. [{"role": "user", "content": "..."}])
        and yields string chunks of the streaming response.
        """
        pass
