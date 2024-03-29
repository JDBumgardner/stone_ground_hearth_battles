from hearthstone.text_agent.text_agent import TextAgentProtocol


class StdIOTransport(TextAgentProtocol):
    """
    Note this agent is blocking, since it uses the same stdin/stdout for all agents.
    """

    async def receive_line(self) -> str:
        return input()

    async def send(self, text: str):
        print(text, end='')
