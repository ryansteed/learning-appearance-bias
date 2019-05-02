import enlighten


class Ticker:
    manager = enlighten.get_manager()

    def __init__(self, total, desc, unit, verbose=False):
        self.total = total
        self.ticker = Ticker.manager.counter(
            total=total,
            desc=desc,
            unit=unit
        )
        self.verbose = verbose
        self.i = 1

    def tick(self):
        if self.verbose:
            print("{}/{}".format(self.i, self.total))
        self.ticker.update()
        self.i += 1

    def update(self):
        self.tick()

    def close(self):
        self.ticker.close()