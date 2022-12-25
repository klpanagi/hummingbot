import asyncio
from typing import TYPE_CHECKING

from hummingbot import init_logging
from hummingbot.client.ui.completer import load_completer
from hummingbot.client.ui.hummingbot_cli import HummingbotCLI
from hummingbot.client.ui.keybindings import load_key_bindings
from hummingbot.client.ui.parser import load_parser
from hummingbot.core.event.events import HummingbotUIEvent
from hummingbot.core.pubsub import PubSub

if TYPE_CHECKING:
    from hummingbot.client.hummingbot_application import HummingbotApplication


class TUIExecutor(HummingbotCLI):
    def __init__(self,
                 hb_app: "HummingbotApplication",
                 *args, **kwargs):
        self.hb_app = hb_app
        command_tabs = self.hb_app.init_command_tabs()
        self.hb_app.parser = load_parser(self.hb_app, command_tabs)
        super().__init__(
            self.hb_app.client_config_map,
            command_tabs=command_tabs,
            input_handler=self.hb_app._handle_command,
            bindings=load_key_bindings(self.hb_app),
            completer=load_completer(self.hb_app),
            **kwargs
        )


class HeadlessExecutor(PubSub):
    def __init__(self,
                 hb_app: "HummingbotApplication"):
        super().__init__()
        # add self.to_stop_config to know if cancel is triggered
        self.to_stop_config: bool = False
        self.live_updates = False
        self.hb_app = hb_app
        # settings
        self.input_event = None

    def clear_input(self):
        """clear_input
        Mock to fake call from ConfigCommand so it does not crash

        Args:

        """
        pass

    def change_prompt(self, prompt: str, is_password: bool = False):
        """change_prompt
        Mock to fake call from so it does not crash

        Args:

        """
        pass

    def did_start(self):
        log_level = self.hb_app.client_config_map.log_level
        init_logging("hummingbot_logs.yml", self.hb_app.client_config_map,
                     override_log_level=log_level)
        self.trigger_event(HummingbotUIEvent.Start, self)

    async def run(self):
        self.did_start()
        if not self.hb_app.ev_loop.is_running():
            self.hb_app.ev_loop.run_forever()
        else:
            while True:
                await asyncio.sleep(0.001)

    def log(self, text: str, save_log: bool = True):
        self.logger().info(text)

    def exit(self):
        pass
