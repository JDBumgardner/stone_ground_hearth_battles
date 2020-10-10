import asyncio
import random
from typing import Optional, Dict, Tuple, Set

from hearthstone.text_agent.lighthouse_speech import LIGHTHOUSE_SPEECH
from hearthstone.text_agent.text_agent import TextAgentProtocol

INSULTS = ['fucker', 'douchebag', 'sweetheart', 'incel', 'son of a hamster', 'foot sniffer', 'proud boy', 'hgh bull',
           'MAGA hatter', 'Shillary', 'sleepy', 'low energy', 'schmeeb', LIGHTHOUSE_SPEECH]

class GameServer:

    def __init__(self, max_sessions: int, kill_event: asyncio.Event):
        self.protocols: Dict[str, 'StoneProtocol'] = {}
        self.player_names: Set[str] = set()
        self.connection_count = 0
        self._max_sessions = max_sessions
        self.kill_event = kill_event
        self.connected = asyncio.Queue()

    def handle_connection(self):
        if self.connection_count < self._max_sessions:
            new_protocol = StoneProtocol(self.kill_event, self.player_names)
            self.connection_count += 1
            self.connected.put_nowait(new_protocol)
            return new_protocol

    async def wait_for_ready(self):
        for _ in range(self._max_sessions):
            session = await self.connected.get()
            await session.ready.wait()
            self.protocols[session.player_name] = session
            print(f'{session.player_name} has joined the game')
            for player in self.protocols.values():
                if player is not session:
                    await player.send(f'the {random.choice(INSULTS)} {session.player_name} has joined the game')
                else:
                    connected_players = "\n".join([f'the {random.choice(INSULTS)} {player_name} has joined already'
                                                   for player_name in self.player_names
                                                   if player_name != player.player_name])
                    await player.send(f'{connected_players}')


class StoneProtocol(asyncio.Protocol, TextAgentProtocol):
    def __init__(self, kill_event: asyncio.Event, player_names: Set[str]):
        self.ready = asyncio.Event()
        self.received_data = False
        self.player_name = None
        self.player_names = player_names
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
        self.data_out.put_nowait(f'welcome {self.peer_address_and_port}, enter your name: ')
        print(f"got connection from {self.peer_address_and_port}")

    def connection_lost(self, exc: Optional[Exception]) -> None:
        self._transport = None
        self.process_task.cancel()
        self.process_task = None
        print(f"lost connection from {self.peer_address_and_port}", exc )

    def data_received(self, data: bytes) -> None:
        if data == b'\xff\xf4\xff\xfd\x06':
            self._transport.close()
            return
        try:
            data = data.decode().rstrip('\n\r')
        except UnicodeDecodeError:
            print(f'unicode decode error {data}')
            return

        self.lines_in.put_nowait(data)
        if not self.received_data:
            if data not in self.player_names:
                self.received_data = True
                self.player_name = data
                self.player_names.add(self.player_name)
                self.ready.set()
            else:
                self.data_out.put_nowait(f'the name {data} has already been chosen by another player')

    async def process_lines(self):
        while True:
            line = await self.data_out.get()
            if 'quit' in line.lower():
                self.kill_event.set()
            self._transport.write((line.rstrip('\r')).encode())

    async def receive_line(self) -> str:
        line = await self.lines_in.get()
        print(f'received the line {line}')
        return line

    async def send(self, data: str):
        self.data_out.put_nowait(data)
