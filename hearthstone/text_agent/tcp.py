import asyncio
from typing import Optional
from hearthstone.text_agent.text_agent import TextAgentTransport


class StoneProtocol(asyncio.Protocol, TextAgentTransport):
    def __init__(self, kill_event: asyncio.Event):
        self.kill_event = kill_event
        self.lines_in = asyncio.Queue()
        self.data_out = asyncio.Queue()
        self.peer_address_and_port = ""
        self._transport: Optional[asyncio.Transport] = None
        self.process_task: Optional[asyncio.Task] = None

    def connection_made(self, transport: asyncio.Transport) -> None:
        self._transport = transport
        self.peer_address_and_port = "%s:%i" % self._transport.get_extra_info('peername')
        self.process_task = asyncio.create_task(self.process_lines())
        self.data_out.put_nowait(f'welcome {self.peer_address_and_port}, enter your name')
        print(f"got connection from {self.peer_address_and_port}")

    def connection_lost(self, exc: Optional[Exception]) -> None:
        self._transport = None
        self.process_task.cancel()
        self.process_task = None
        print(f"lost connection from {self.peer_address_and_port}", exc )

    def data_received(self, data: bytes) -> None:
        for line in data.split(b'\n'):
            self.lines_in.put_nowait(line.decode().rstrip() + '\n')

    async def process_lines(self):
        while True:
            line = await self.data_out.get()
            if 'quit' in line.lower():
                self.kill_event.set()
            for l in line.split('\n'):
                self._transport.write(l.encode() + b'\n')

    async def receive_line(self) -> str:
        return await self.lines_in.get()

    async def send(self, data: str):
        self.data_out.put_nowait(data)
