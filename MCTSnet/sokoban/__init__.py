import logging
import pkg_resources
import json
from gym.envs.registration import register
from . import solver


logger = logging.getLogger(__name__)

resource_package = __name__
env_json = pkg_resources.resource_filename(resource_package, 'available_envs.json')

with open(env_json) as f:
    envs = json.load(f)
    for en in envs:
        register(
            id=en["id"],
            entry_point=en["entry_point"]
        )
