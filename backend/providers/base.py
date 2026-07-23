from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional, Union, Dict, Any
from api.chat.schemas import Message

class BaseProvider(ABC):
    @abstractmethod
    async def stream_response(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Accepts a list of Message objects (and optional tools)
        and yields string chunks of the streaming response or 
        a dictionary containing intercepted tool calls.
        """
        pass
