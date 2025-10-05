import numpy as np
import gymnasium as gym
from gymnasium import spaces

import os, math, random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np

try:
    import pygame
except Exception:
    pygame = None

from PIL import Image, ImageDraw

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
DRONE_PATH = os.path.join(ASSETS_DIR, "drone.png")
BUILDING_PATH = os.path.join(ASSETS_DIR, "building.png")

def _ensure_assets():
    """Create simple placeholder icons if missing."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    if not os.path.exists(DRONE_PATH):
        img = Image.new("RGBA", (256,256), (0,0,0,0))
        d = ImageDraw.Draw(img)
        d.ellipse((60,60,196,196), fill=(90,90,90,200), outline=(20,20,20,255), width=8)
        d.rectangle((120,24,136,232), fill=(20,20,20,180))
        d.rectangle((24,120,232,136), fill=(20,20,20,180))
        img.save(DRONE_PATH)
    if not os.path.exists(BUILDING_PATH):
        img = Image.new("RGBA", (256,256), (30,30,35,255))
        d = ImageDraw.Draw(img)
        d.rectangle((40,40,216,216), fill=(60,60,70,255), outline=(20,20,25,255), width=8)
        # windows
        for r in range(5):
            for c in range(4):
                x0 = 60 + c*36
                y0 = 60 + r*30
                d.rectangle((x0, y0, x0+20, y0+18), fill=(245,223,77,255))
        img.save(BUILDING_PATH)

def _try_load_sprite(path, size):
    if pygame is None:
        return None
    try:
        surf = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(surf, (size,size))
    except Exception:
        return None

@dataclass
class GridWorldConfig:
    grid_size: int = 10
    num_agents: int = 5
    max_steps: int = 300
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    render_fps: int = 30

    # multi-world (federated) layout
    worlds: int = 1


class FederatedSingleEnv(gym.Env):
    """Single grid world with multiple agents (drones) and moving buildings as goals."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str = None, **kwargs):
        super().__init__()
        self.cfg = GridWorldConfig(
            grid_size=int(kwargs.get("grid_size", 10)),
            num_agents=int(kwargs.get("num_agents", 5)),
            max_steps=int(kwargs.get("max_steps", 300)),
            step_penalty=float(kwargs.get("step_penalty", -0.01)),
            goal_reward=float(kwargs.get("goal_reward", 1.0)),
            render_fps=int(kwargs.get("render_fps", 30)),
        )
        self.render_mode = render_mode
        self._seed = None

        # Action/Obs spaces
        self.action_space = spaces.MultiDiscrete([5]*self.cfg.num_agents)  # 0=stay,1=up,2=down,3=left,4=right
        # obs = concat of agent positions and goal positions, normalized to [0,1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4*self.cfg.num_agents,), dtype=np.float32
        )

        # State
        self._agents = None
        self._goals = None
        self._steps = 0

        # Rendering
        self._window = None
        self._clock = None
        self._cell = 56  # cell pixel size
        self._drone_sprites = None
        self._building_sprite = None

        _ensure_assets()

    # ---------------- Gymnasium API ----------------
    def seed(self, seed=None):
        self._seed = seed
        rng = np.random.default_rng(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        gs = self.cfg.grid_size
        self._steps = 0
        # random positions: (num_agents, 2)
        self._agents = np.random.randint(0, gs, size=(self.cfg.num_agents, 2), dtype=np.int32)
        self._goals  = np.random.randint(0, gs, size=(self.cfg.num_agents, 2), dtype=np.int32)

        obs = self._get_obs()
        info = {}

        # set up rendering
        if self.render_mode == "human":
            self._ensure_window()

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        gs = self.cfg.grid_size
        self._steps += 1

        # Apply actions
        for i, a in enumerate(action):
            if a == 1:   # up
                self._agents[i,1] = max(0, self._agents[i,1]-1)
            elif a == 2: # down
                self._agents[i,1] = min(gs-1, self._agents[i,1]+1)
            elif a == 3: # left
                self._agents[i,0] = max(0, self._agents[i,0]-1)
            elif a == 4: # right
                self._agents[i,0] = min(gs-1, self._agents[i,0]+1)
            # else 0=stay

        # Rewards: per agent +1 if reaches its goal; goal respawns
        reward = 0.0
        for i in range(self.cfg.num_agents):
            if (self._agents[i] == self._goals[i]).all():
                reward += self.cfg.goal_reward
                self._goals[i] = np.random.randint(0, gs, size=(2,), dtype=np.int32)

        # Step penalty
        reward += self.cfg.step_penalty * self.cfg.num_agents

        terminated = False  # no absorbing terminal; episodic via truncation
        truncated = self._steps >= self.cfg.max_steps
        info = {}

        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            # support headless collection by calling render (optional)
            pass

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_array()

    def close(self):
        if self._window is not None and pygame is not None:
            pygame.display.quit()
            pygame.quit()
        self._window = None
        self._clock = None

    # ---------------- internals ----------------
    def _get_obs(self):
        gs = float(self.cfg.grid_size-1 if self.cfg.grid_size>1 else 1)
        data = np.concatenate([self._agents.flatten(), self._goals.flatten()]).astype(np.float32)
        data /= (gs)
        return data

    def _ensure_window(self):
        if pygame is None:
            return
        if self._window is None:
            pygame.init()
            w = h = self.cfg.grid_size * self._cell
            self._window = pygame.display.set_mode((w, h))
            pygame.display.set_caption("FederatedSingleEnv")
            self._clock = pygame.time.Clock()
            # load sprites
            pad = max(0, int(self._cell*0.2))
            size = max(8, self._cell - pad)
            self._building_sprite = _try_load_sprite(BUILDING_PATH, size)
            self._drone_sprites = [_try_load_sprite(DRONE_PATH, size) for _ in range(self.cfg.num_agents)]

    def _render_human(self):
        if pygame is None:
            return
        self._ensure_window()
        w = h = self.cfg.grid_size * self._cell
        surf = pygame.Surface((w, h))
        surf.fill((240, 240, 242))
        # grid
        for i in range(self.cfg.grid_size+1):
            pygame.draw.line(surf, (200,200,210), (0, i*self._cell), (w, i*self._cell), 1)
            pygame.draw.line(surf, (200,200,210), (i*self._cell, 0), (i*self._cell, h), 1)
        # draw goals (buildings)
        for g in self._goals:
            x, y = g
            rect = pygame.Rect(x*self._cell, y*self._cell, self._cell, self._cell)
            if self._building_sprite is not None:
                blit_rect = self._building_sprite.get_rect(center=rect.center)
                surf.blit(self._building_sprite, blit_rect)
            else:
                pygame.draw.rect(surf, (60,60,80), rect.inflate(-12,-12), border_radius=10)
        # draw agents (drones)
        for i,a in enumerate(self._agents):
            x, y = a
            rect = pygame.Rect(x*self._cell, y*self._cell, self._cell, self._cell)
            if self._drone_sprites[i] is not None:
                blit_rect = self._drone_sprites[i].get_rect(center=rect.center)
                surf.blit(self._drone_sprites[i], blit_rect)
            else:
                pygame.draw.circle(surf, (40,120,220), rect.center, self._cell//3)
        # flip
        self._window.blit(surf, (0,0))
        pygame.display.flip()
        if self._clock:
            self._clock.tick(self.cfg.render_fps)

    def _render_array(self):
        # Fallback lightweight array rendering without pygame
        # Draw a simple RGB array (grid + agents/goals)
        gs = self.cfg.grid_size
        cell = 16
        H = W = gs*cell
        img = np.ones((H,W,3), dtype=np.uint8) * 245
        # grid
        for i in range(gs+1):
            img[i*cell-1:i*cell+1,:,:] = 200
            img[:, i*cell-1:i*cell+1,:] = 200
        # goals (dark)
        for gx,gy in self._goals:
            y0, x0 = gy*cell, gx*cell
            img[y0+2:y0+cell-2, x0+2:x0+cell-2] = (60,60,80)
        # agents (blue-ish)
        for ax,ay in self._agents:
            cy, cx = ay*cell + cell//2, ax*cell + cell//2
            rr = cell//3
            y, x = np.ogrid[-rr:rr, -rr:rr]
            mask = x*x + y*y <= rr*rr
            y0, x0 = cy-rr, cx-rr
            img[y0:y0+2*rr, x0:x0+2*rr][mask] = (40,120,220)
        return img
