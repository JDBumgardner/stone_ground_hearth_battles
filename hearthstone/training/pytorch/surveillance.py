class GlobalStepContext:
    def get_global_step(self) -> int:
        raise NotImplemented("Not Implemented")

    def should_plot(self) -> bool:
        raise NotImplemented("Not Implemented")
