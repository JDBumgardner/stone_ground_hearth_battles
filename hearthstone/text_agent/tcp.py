import asyncio
import random
import ipaddress
from typing import Optional, Dict, Tuple, Set
from aioupnp.upnp import UPnP, UPnPError
from hearthstone.text_agent.lighthouse_speech import LIGHTHOUSE_SPEECH
from hearthstone.text_agent.text_agent import TextAgentProtocol


INSULTS = ['fucker', 'douchebag', 'sweetheart', 'incel', 'son of a hamster', 'foot sniffer', 'proud boy', 'hgh bull',
           'MAGA hatter', 'Shillary', 'sleepy', 'low energy', 'schmeeb', LIGHTHOUSE_SPEECH]

TELNET_CTRL_C = b'\xff\xf4\xff\xfd\x06'


CARRIER_GRADE_NAT_SUBNET = ipaddress.ip_network('100.64.0.0/10')
IPV4_TO_6_RELAY_SUBNET = ipaddress.ip_network('192.88.99.0/24')


def is_valid_public_ipv4(address):
    try:
        parsed_ip = ipaddress.ip_address(address)
        if any((parsed_ip.version != 4, parsed_ip.is_unspecified, parsed_ip.is_link_local, parsed_ip.is_loopback,
                parsed_ip.is_multicast, parsed_ip.is_reserved, parsed_ip.is_private, parsed_ip.is_reserved)):
            return False
        else:
            return not any((CARRIER_GRADE_NAT_SUBNET.supernet_of(ipaddress.ip_network(f"{address}/32")),
                            IPV4_TO_6_RELAY_SUBNET.supernet_of(ipaddress.ip_network(f"{address}/32"))))
    except (ipaddress.AddressValueError, ValueError):
        return False


class GameServer:
    def __init__(self, max_sessions: int, kill_event: asyncio.Event):
        self.protocols: Dict[str, 'StoneProtocol'] = {}
        self.player_names: Set[str] = set()
        self.connection_count = 0
        self._max_sessions = max_sessions
        self.kill_event = kill_event
        self.connected = asyncio.Queue()
        self._serve_task: Optional[asyncio.Task] = None

    def handle_connection(self):
        if self.connection_count < self._max_sessions:
            new_protocol = StoneProtocol(self.kill_event, self.player_names, self.disconnect)
            self.connection_count += 1
            self.connected.put_nowait(new_protocol)
            return new_protocol

    async def wait_for_ready(self):
        while len(self.protocols) < self._max_sessions:
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

    def disconnect(self, session: 'StoneProtocol'):
        if session.player_name and session.player_name in self.protocols:
            self.protocols.pop(session.player_name)
        self.connection_count -= 1

    def serve_forever(self, interface: str = '0.0.0.0', port: int = 9998):
        async def serve_forever():
            set_port_mapping = False
            u = None
            try:
                u = await UPnP.discover()
            except UPnPError:
                print("failed to set up port redirect with UPnP... proceeding anyway")
                mapped_port = port
            else:
                external_ip = await u.get_external_ip()
                if not is_valid_public_ipv4(external_ip):
                    raise Exception("failed to get valid external address from UPnP, are you behind a double NAT?")
                mapped_port = await u.get_next_mapping(port, 'TCP', 'cyborg arena')
                set_port_mapping = True
                print("set port mapping")
            try:
                server = await asyncio.get_event_loop().create_server(self.handle_connection, interface, mapped_port)
                print(f"starting server on {interface}:{mapped_port}")
                async with server:
                    await server.serve_forever()
            finally:
                if set_port_mapping and u:
                    await u.delete_port_mapping(mapped_port, 'TCP')
                    print("deleted port mapping")

        if self._serve_task:
            raise Exception("fnord")

        self._serve_task = asyncio.create_task(serve_forever())


class StoneProtocol(asyncio.Protocol, TextAgentProtocol):
    def __init__(self, kill_event: asyncio.Event, player_names: Set[str], disconnect_fn):
        self._disconnect = disconnect_fn
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
        self._connection_lost = asyncio.Event()

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
        print(f"lost connection from {self.peer_address_and_port}", exc)
        self._disconnect(self)
        self._connection_lost.set()

    def data_received(self, data: bytes) -> None:
        if data == TELNET_CTRL_C:
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
        line_task = asyncio.create_task(self.lines_in.get())
        connection_lost_task = asyncio.create_task(self._connection_lost.wait())
        await asyncio.wait([line_task, connection_lost_task], return_when=asyncio.FIRST_COMPLETED)
        if connection_lost_task.done():
            raise ConnectionError
        line = await line_task
        print(f'received the line {line}')
        return line

    async def send(self, data: str):
        self.data_out.put_nowait(data)
