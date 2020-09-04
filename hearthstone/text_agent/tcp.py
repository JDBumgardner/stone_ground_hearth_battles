import trio

from hearthstone.agent import Agent
from hearthstone.text_agent.line_reader import LineReader
from hearthstone.text_agent.text_agent import TextAgentTransport


class TcpTransport(TextAgentTransport):
    """
    Non-blocking Text agent that communicates over TCP.
    """
    def __init__(self, stream: trio.abc.Stream):
        self.stream = stream
        self.line_reader = LineReader(self.stream, 1024)

    async def receive_line(self) -> str:
        return (await self.line_reader.readline()).decode('utf-8').rstrip()

    async def send(self, text: str):
        await self.stream.send_all(text.encode('utf-8'))
