import asyncio

from hearthstone.text_agent.text_agent import TextAgentTransport


class TcpTransport(TextAgentTransport):
    """
    Non-blocking Text agent that communicates over TCP.
    """
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer

    async def receive_line(self) -> str:
        try:
            return (await self.reader.readline()).decode('utf-8').rstrip()
        except UnicodeDecodeError:
            print("fnord")
            return ""

    async def send(self, text: str):
        self.writer.write(text.encode('utf-8'))
        await self.writer.drain()
