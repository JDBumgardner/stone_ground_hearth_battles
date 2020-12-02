import asyncio


def get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as e:
        asyncio.set_event_loop(asyncio.new_event_loop())
        return asyncio.get_event_loop()