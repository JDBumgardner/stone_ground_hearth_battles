from hearthstone.text_agent.text_agent import TextAgentTransport


class StdIOTransport(TextAgentTransport):
    """
    Note this agent is blocking, since it uses the same stdin/stdout for all agents.
    """
    async def receive_line(self) -> str:
        return input()

    async def send(self, text: str):
        print(text, end='')
