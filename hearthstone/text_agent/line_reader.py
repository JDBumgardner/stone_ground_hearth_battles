# https://stackoverflow.com/a/53576829/94102
class LineReader:
    def __init__(self, stream, max_line_length=16384):
        self.stream = stream
        self._line_generator = self.generate_lines(max_line_length)

    @staticmethod
    def generate_lines(max_line_length):
        buf = bytearray()
        find_start = 0
        while True:
            newline_idx = buf.find(b'\n', find_start)
            if newline_idx < 0:
                # no b'\n' found in buf
                if len(buf) > max_line_length:
                    raise ValueError("line too long")
                # next time, start the search where this one left off
                find_start = len(buf)
                more_data = yield
            else:
                # b'\n' found in buf so return the line and move up buf
                line = buf[:newline_idx + 1]
                # Update the buffer in place, to take advantage of bytearray's
                # optimized delete-from-beginning feature.
                del buf[:newline_idx + 1]
                # next time, start the search from the beginning
                find_start = 0
                more_data = yield line

            if more_data is not None:
                buf += bytes(more_data)

    async def readline(self):
        line = next(self._line_generator)
        while line is None:
            more_data = await self.stream.receive_some(1024)
            if not more_data:
                return b''  # this is the EOF indication expected by my caller
            line = self._line_generator.send(more_data)
        return line
