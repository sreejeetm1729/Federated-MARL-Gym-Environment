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


class FederatedMultiEnv(gym.Env):
    """Multiple independent grid worlds in parallel (batched).

    Actions are per-world multi-agent MultiDiscrete vectors. The step returns
    aggregated reward (sum across worlds). Useful for simulating federated updates.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str = None, **kwargs):
        super().__init__()
        worlds = int(kwargs.get("worlds", 4))
        num_agents = int(kwargs.get("num_agents", 5))
        grid_size = int(kwargs.get("grid_size", 10))

        self.cfg = GridWorldConfig(
            grid_size=grid_size,
            num_agents=num_agents,
            max_steps=int(kwargs.get("max_steps", 250)),
            step_penalty=float(kwargs.get("step_penalty", -0.01)),
            goal_reward=float(kwargs.get("goal_reward", 1.0)),
            render_fps=int(kwargs.get("render_fps", 30)),
            worlds=worlds,
        )
        self.render_mode = render_mode

        # action: per world, a MultiDiscrete([5]*num_agents)
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([5]*num_agents) for _ in range(worlds)])
        # observation: for each world, concatenation of agents+goals normalized
        self.observation_space = spaces.Tuple([
            spaces.Box(low=0.0, high=1.0, shape=(4*num_agents,), dtype=np.float32) for _ in range(worlds)
        ])

        # State
        self._agents = None  # shape (worlds, num_agents, 2)
        self._goals = None   # shape (worlds, num_agents, 2)
        self._steps = 0

        # Rendering
        self._window = None
        self._clock = None
        self._cell = 48
        self._drone_sprite = None
        self._building_sprite = None

        _ensure_assets()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        gs = self.cfg.grid_size
        W, A = self.cfg.worlds, self.cfg.num_agents
        self._steps = 0
        self._agents = np.random.randint(0, gs, size=(W, A, 2), dtype=np.int32)
        self._goals  = np.random.randint(0, gs, size=(W, A, 2), dtype=np.int32)

        if self.render_mode == "human":
            self._ensure_window()

        return self._get_obs_tuple(), {}  # info empty

    def step(self, action):
        # action is a tuple of length W, each is array-like of length A
        gs = self.cfg.grid_size
        self._steps += 1
        W, A = self.cfg.worlds, self.cfg.num_agents

        total_reward = 0.0
        for w in range(W):
            aw = np.asarray(action[w], dtype=np.int64)
            for i, a in enumerate(aw):
                if a == 1:   # up
                    self._agents[w,i,1] = max(0, self._agents[w,i,1]-1)
                elif a == 2: # down
                    self._agents[w,i,1] = min(gs-1, self._agents[w,i,1]+1)
                elif a == 3: # left
                    self._agents[w,i,0] = max(0, self._agents[w,i,0]-1)
                elif a == 4: # right
                    self._agents[w,i,0] = min(gs-1, self._agents[w,i,0]+1)
            # reward for world w
            for i in range(A):
                if (self._agents[w,i] == self._goals[w,i]).all():
                    total_reward += self.cfg.goal_reward
                    self._goals[w,i] = np.random.randint(0, gs, size=(2,), dtype=np.int32)
            total_reward += self.cfg.step_penalty * A

        terminated = False
        truncated = self._steps >= self.cfg.max_steps
        info = {}

        if self.render_mode == "human":
            self._render_human()
        return self._get_obs_tuple(), total_reward, terminated, truncated, info

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

    # --------------- internals ---------------
    def _get_obs_tuple(self):
        gs = float(self.cfg.grid_size-1 if self.cfg.grid_size>1 else 1)
        obs = []
        for w in range(self.cfg.worlds):
            data = np.concatenate([self._agents[w].flatten(), self._goals[w].flatten()]).astype(np.float32)
            data /= gs
            obs.append(data)
        return tuple(obs)

    def _ensure_window(self):
        if pygame is None:
            return
        if self._window is None:
            pygame.init()
            # tile W worlds horizontally
            w = self.cfg.worlds * self.cfg.grid_size * self._cell
            h = self.cfg.grid_size * self._cell
            self._window = pygame.display.set_mode((w, h))
            pygame.display.set_caption("FederatedMultiEnv")
            self._clock = pygame.time.Clock()
            pad = max(0, int(self._cell*0.25))
            size = max(8, self._cell - pad)
            self._building_sprite = _try_load_sprite(BUILDING_PATH, size)
            self._drone_sprite = _try_load_sprite(DRONE_PATH, size)

    def _render_human(self):
        if pygame is None:
            return
        self._ensure_window()
        gw = self.cfg.grid_size * self._cell
        h = self.cfg.grid_size * self._cell
        surf = pygame.Surface((gw*self.cfg.worlds, h))
        surf.fill((245,245,248))

        # draw each world
        for w in range(self.cfg.worlds):
            x_off = w*gw
            # grid
            for i in range(self.cfg.grid_size+1):
                pygame.draw.line(surf, (200,200,210), (x_off, i*self._cell), (x_off+gw, i*self._cell), 1)
                pygame.draw.line(surf, (200,200,210), (x_off+i*self._cell, 0), (x_off+i*self._cell, h), 1)
            # goals
            for g in self._goals[w]:
                x, y = g
                rect = pygame.Rect(x_off + x*self._cell, y*self._cell, self._cell, self._cell)
                if self._building_sprite is not None:
                    blit_rect = self._building_sprite.get_rect(center=rect.center)
                    surf.blit(self._building_sprite, blit_rect)
                else:
                    pygame.draw.rect(surf, (60,60,80), rect.inflate(-12,-12), border_radius=8)
            # agents
            for a in self._agents[w]:
                x, y = a
                rect = pygame.Rect(x_off + x*self._cell, y*self._cell, self._cell, self._cell)
                if self._drone_sprite is not None:
                    blit_rect = self._drone_sprite.get_rect(center=rect.center)
                    surf.blit(self._drone_sprite, blit_rect)
                else:
                    pygame.draw.circle(surf, (40,120,220), rect.center, self._cell//3)

        self._window.blit(surf, (0,0))
        pygame.display.flip()
        if self._clock:
            self._clock.tick(self.cfg.render_fps)

    def _render_array(self):
        # Concatenate worlds horizontally
        frames = []
        for w in range(self.cfg.worlds):
            # reuse the single-env array renderer trick
            gs = self.cfg.grid_size
            cell = 12
            H = gs*cell
            W = gs*cell
            img = np.ones((H,W,3), dtype=np.uint8) * 245
            # grid
            for i in range(gs+1):
                img[i*cell-1:i*cell+1,:,:] = 200
                img[:, i*cell-1:i*cell+1,:] = 200
            # goals
            for gx,gy in self._goals[w]:
                y0, x0 = gy*cell, gx*cell
                img[y0+2:y0+cell-2, x0+2:x0+cell-2] = (60,60,80)
            # agents
            for ax,ay in self._agents[w]:
                cy, cx = ay*cell + cell//2, ax*cell + cell//2
                rr = cell//3
                y, x = np.ogrid[-rr:rr, -rr:rr]
                mask = x*x + y*y <= rr*rr
                y0, x0 = cy-rr, cx-rr
                img[y0:y0+2*rr, x0:x0+2*rr][mask] = (40,120,220)
            frames.append(img)
        return np.concatenate(frames, axis=1)
