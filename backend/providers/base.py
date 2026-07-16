from abc import ABC, abstractmethod
from typing import AsyncGenerator, List
from api.chat.schemas import Message

class BaseProvider(ABC):
    @abstractmethod
    async def stream_response(self, messages: List[Message]) -> AsyncGenerator[str, None]:
        """
        Accepts a list of Message objects
        and yields string chunks of the streaming response.
        """
        pass
