from dsl_manager.json_dsl_manager import JsonDslManager
from console_manager.console_manager import ConsoleManager


class IISNeRF:

    def __init__(self):
        self.console_manager = ConsoleManager()
        self.args = self.console_manager.parse_args()
        self.json_dsl_manager = JsonDslManager()

    def run(self):
        if self.args.nerf_variant and self.args.scenario and self.args.scenario_path:
            if self.args.dataset_type:
                self.json_dsl_manager.run_single_scenario(self.args.nerf_variant, self.args.scenario,
                                                          self.args.scenario_path,
                                                          self.args.dataset_type)
            else:
                self.json_dsl_manager.run_single_scenario(self.args.nerf_variant, self.args.scenario,
                                                          self.args.scenario_path)


if __name__ == "__main__":
    iis_nerf = IISNeRF()
    iis_nerf.run()
