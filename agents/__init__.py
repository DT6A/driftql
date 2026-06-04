import importlib
from collections.abc import Mapping


_AGENT_SPECS = {
    'diffusion_ql': ('agents.diffusion_ql', 'DiffusionQLAgent'),
    'driftql': ('agents.driftql', 'DriftQLAgent'),
    'fql': ('agents.fql', 'FQLAgent'),
    'idql': ('agents.idql', 'IDQLAgent'),
    'ifql': ('agents.ifql', 'IFQLAgent'),
    'iql': ('agents.iql', 'IQLAgent'),
    'rebrac': ('agents.rebrac', 'ReBRACAgent'),
    'sac': ('agents.sac', 'SACAgent'),
}


class AgentRegistry(Mapping):
    """Mapping from agent name to class, loaded on first use."""

    def __init__(self, specs):
        self._specs = specs
        self._cache = {}

    def __getitem__(self, name):
        if name not in self._specs:
            raise KeyError(name)
        if name not in self._cache:
            module_name, class_name = self._specs[name]
            module = importlib.import_module(module_name)
            self._cache[name] = getattr(module, class_name)
        return self._cache[name]

    def __iter__(self):
        return iter(self._specs)

    def __len__(self):
        return len(self._specs)


agents = AgentRegistry(_AGENT_SPECS)
