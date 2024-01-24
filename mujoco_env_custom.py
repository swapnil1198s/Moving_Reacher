from os import path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, logger, spaces
from gymnasium.spaces import Space

#mujoco_py is not compatible with newer versions of MuJoCo so this version only uses the mujoco library
try:
     import mujoco
except ImportError as e:
     MUJOCO_IMPORT_ERROR = e
else:
     MUJOCO_IMPORT_ERROR = None

#Default screen size
DEFAULT_SIZE = 800

def expand_model_path(model_path: str) -> str:
    """Expands the `model_path` to a full path if it starts with '~' or '.' or '/'."""
    if model_path.startswith(".") or model_path.startswith("/"):
        fullpath = model_path
    elif model_path.startswith("~"):
        fullpath = path.expanduser(model_path)
    else:
        fullpath = path.join(path.dirname(__file__), "assets", model_path)
    if not path.exists(fullpath):
        raise OSError(f"File {fullpath} does not exist")

    return fullpath

class MujocoSim(gym.Env[NDArray[np.float64], NDArray[np.float32]]):
    def __init__(
        self,
        model_path,
        frame_skip,
        observation_space: Optional[Space], #Optional[X] is equivalent to Union[X, None].
        render_mode: Optional[str]=None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int]=None,
        camera_name: Optional[str]=None,
        default_camera_config: Optional[Dict[str, Union[float, int]]] = None,
        max_geom: int = 1000,
    ):
        # TODO: Decide on the ommition or imclusion of camera_id and camera_name
        """
        model_path: Path to the MuJoCo Model.
        frame_skip: Number of MuJoCo simulation steps per gym `step()`.
        observation_space: The observation space of the environment.
        render_mode: The `render_mode` used.
        width: The width of the render window.
        height: The height of the render window.
        camera_id: The camera ID used.
        camera_name: The name of the camera used (can not be used in conjunction with `camera_id`).
        """
        self.full_model_path = expand_model_path(model_path)

        #window dimensions
        self.width = width
        self.height= height
        
        # Load up the mujoco model of reacher and the data associated with it.
        self.model, self.data = self._init_mujoco_sim()

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.frame_skip = frame_skip
        ##TODO: need to add action space prop
        self.render_mode = render_mode
        self.observation_space = observation_space
        self._set_action_space()
        
        from mujoco_rendering_custom import MujocoSimRenderer
        self.renderer = MujocoSimRenderer(
            self.model,
            self.data,
            default_camera_config,
            self.width,
            self.height,
            max_geom
        )
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space
    
    def _get_reset_info(self) -> Dict[str, float]:
        """Function that generates the `info` that is returned during a `reset()`."""
        return {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self._reset_sim()

        ob = self.reset_model()
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()
        return ob, info


    def _init_mujoco_sim(self) -> Tuple["mujoco._structs.MjModel", "mujoco._structs.MjData"]:
        model = mujoco.MjModel.from_xml_path(self.full_model_path)
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

    def _reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        
        mujoco.mj_forward(self.model, self.data)

    def do_simulation(self, ctrl, n_frames) -> None:
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_mujoco_sim(ctrl, n_frames)

    def _step_mujoco_sim(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep = n_frames)
        mujoco.mj_rnePostConstraint(self.model, self.data)
    
    def render(self):
        return self.renderer.render(
            self.render_mode
        )
    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos