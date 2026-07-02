import queue
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(self, data_handler, strategy, portfolio, execution):
        self.data = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.events: queue.Queue = queue.Queue()

        # wire shared event queue into each component
        self.strategy.events = self.events
        self.portfolio.events = self.events
        self.execution.events = self.events

    def run(self) -> None:
        data_stream = self.data.stream()
        data_exhausted = False
        bars_processed = 0

        while True:
            if self.events.empty():
                if data_exhausted:
                    break
                try:
                    market_event = next(data_stream)
                    self.events.put(market_event)
                    bars_processed += 1
                except StopIteration:
                    data_exhausted = True
                continue

            event = self.events.get(block=False)

            if event.type == "MARKET":
                self.strategy.calculate_signals(event)
                self.portfolio.update_timeindex(event)

            elif event.type == "SIGNAL":
                self.portfolio.update_signal(event)

            elif event.type == "ORDER":
                self.execution.execute_order(event)

            elif event.type == "FILL":
                self.portfolio.update_fill(event)

        logger.info("Backtest complete. Bars processed: %d", bars_processed)
