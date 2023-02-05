import time
from typing import TYPE_CHECKING, Callable

from hummingbot.core.utils.async_utils import safe_ensure_future

if TYPE_CHECKING:
    from hummingbot.client.hummingbot_application import HummingbotApplication


class PnLResetCommand:
    """PnLResetCommand.
    """

    def pnl_reset(self,  # type: HummingbotApplication
                  callback: Callable = None):
        """pnl_reset.

        Resets the bot's init time to current time.

        """
        task = safe_ensure_future(
            self.pnl_reset_async(),
            loop=self.ev_loop
        )
        if callback is not None:
            task.add_done_callback(callback)
        return task

    async def pnl_reset_async(self,  # type: HummingbotApplication
                              ):
        """pnl_reset_async.

        Resets the bot's init time to current time.

        """
        self.init_time = time.time()
        # trade_monitor only updates when the trades are more than 0, thus setting it here.
        self.app.trade_monitor.log("Trades: 0, Total PnL: 0.00, Return %: 0.00%")
        self.logger().info("PnL init timestamp has been reset")
        self.notify('PnL has been reset')
