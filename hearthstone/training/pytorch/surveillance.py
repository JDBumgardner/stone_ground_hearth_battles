class GlobalStepContext:
    def get_global_step(self) -> int:
        raise NotImplemented

    def should_plot(self) -> bool:
        raise NotImplemented
