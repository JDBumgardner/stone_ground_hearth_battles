class TripleRewardCard:
    def __init__(self, level: int):
        self.level = level

    def __repr__(self):
        return "tier " + str(self.level)
