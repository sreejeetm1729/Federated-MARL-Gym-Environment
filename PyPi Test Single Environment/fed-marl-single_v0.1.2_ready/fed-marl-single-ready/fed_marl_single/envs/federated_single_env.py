# === EDITED FOR PACKAGING ===
# - Accepts kwargs from gym.make: grid_size / width / height / num_agents / max_steps / seed / render_mode, etc.
# - Removes any auto-execution on import (no training loops kick off automatically).
# - Provides a stable class name `FederatedSingleEnv` that Gymnasium entry-point can target.
#
# Usage:
#   import gymnasium as gym, fed_marl_single
#   env = gym.make("FederatedSingle-v0", grid_size=10, num_agents=4, render_mode="rgb_array")

from __future__ import annotations
import os, random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

# Prefer gymnasium; raise helpful error if missing
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError("Please install gymnasium: pip install gymnasium") from e

import numpy as np
from PIL import Image, ImageDraw

# Optional dependencies (only needed for video recording functionality)
try:
    import pygame
except Exception as e:
    raise SystemExit("Please install pygame: pip install pygame") from e

try:
    import imageio.v2 as imageio  # optional
except Exception:
    imageio = None

# ---------------- assets: ensure placeholders if missing ----------------
ASSETS_DIR = "assets"
DRONE_PATH = os.path.join(ASSETS_DIR, "drone.png")
BUILDING_PATH = os.path.join(ASSETS_DIR, "building.png")

def ensure_assets():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    if not os.path.exists(DRONE_PATH):
        img = Image.new("RGBA", (256,256), (0,0,0,0))
        d = ImageDraw.Draw(img)
        d.ellipse((88,88,168,168), fill=(180,180,185,255))
        d.rectangle((126,20,130,236), fill=(160,160,165,255))
        d.rectangle((20,126,236,130), fill=(160,160,165,255))
        for cx,cy in [(128,32),(128,224),(32,128),(224,128)]:
            d.ellipse((cx-24,cy-24,cx+24,cy+24), outline=(90,90,95,255), width=6, fill=(210,210,215,255))
        img.save(DRONE_PATH)
    if not os.path.exists(BUILDING_PATH):
        img = Image.new("RGBA", (256,256), (0,0,0,0))
        d = ImageDraw.Draw(img)
        d.rectangle((70,60,186,210), fill=(70,120,200,255))
        d.rectangle((90,30,166,60), fill=(60,100,180,255))
        for r in range(4):
            for c in range(3):
                x0 = 82 + c*28; y0 = 74 + r*32
                d.rectangle((x0,y0,x0+20,y0+20), fill=(230,240,255,255))
        d.rectangle((122,170,134,210), fill=(40,70,130,255))
        img.save(BUILDING_PATH)

ensure_assets()

# ---------------- Recording helper ----------------
class PygameRecorder:
    def __init__(self, path: str, fps: int = 8):
        if imageio is None:
            raise RuntimeError("imageio is not installed. Install with: pip install imageio imageio-ffmpeg")
        self.fps = int(fps)
        base, ext = os.path.splitext(path)
        ext_lower = ext.lower()
        if ext_lower not in (".mp4", ".gif"):
            ext_lower = ".mp4"
            path = base + ext_lower
        self.path = path
        self.ext = ext_lower
        self.writer = None

    def _open_writer(self, frame_shape_hw3):
        if self.ext == ".mp4":
            try:
                self.writer = imageio.get_writer(self.path, fps=self.fps, codec="libx264", quality=8)
                return
            except Exception:
                self.ext = ".gif"
                self.path = os.path.splitext(self.path)[0] + ".gif"
        if self.ext == ".gif":
            self.writer = imageio.get_writer(self.path, format="GIF", mode="I", duration=1.0/max(1,self.fps), loop=0)

    def append(self, surface):
        import pygame as _pg
        arr = _pg.surfarray.array3d(surface)
        frame = np.transpose(arr, (1,0,2))
        if self.writer is None:
            self._open_writer(frame.shape)
        self.writer.append_data(frame)

    def close(self):
        if self.writer is not None:
            self.writer.close()

# ---------------- Core config & env ----------------
@dataclass
class GridConfig:
    width: int = 7
    height: int = 7
    num_agents: int = 5
    max_steps: int = 160
    step_penalty: float = 0.01
    goal_reward: float = 1.0
    collision_penalty: float = 0.05
    seed: Optional[int] = 42

class MultiAgentGridEnv(gym.Env):
    metadata = {"render_modes": ["human","rgb_array"], "render_fps": 8}

    def __init__(self, cfg: GridConfig, render_mode: Optional[str] = "human",
                 watermark_text: Optional[str] = "sreejeetm1729", recorder: Optional[PygameRecorder] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.rng = np.random.RandomState(cfg.seed)

        self.action_space = spaces.MultiDiscrete([5]*cfg.num_agents)
        obs_dim = cfg.num_agents * 2 * 2
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        self._agent_pos: List[Tuple[int,int]] = []
        self._goals: List[Tuple[int,int]] = []
        self._t = 0
        self._last_collisions = 0

        # pygame stuff
        self._viewer = None
        self._surface = None
        self._clock = None
        self._font = None
        self._drone_sprites = None
        self._building_sprite = None

        # watermark & recorder
        self._watermark_text = watermark_text
        self._recorder = recorder

    def set_recorder(self, recorder: Optional[PygameRecorder]):
        self._recorder = recorder

    def set_watermark(self, text: Optional[str]):
        self._watermark_text = text

    def seed(self, seed: Optional[int] = None):
        self.cfg.seed = seed
        self.rng = np.random.RandomState(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        self._t = 0
        self._last_collisions = 0
        self._agent_pos = []
        occ = set()
        while len(self._agent_pos) < self.cfg.num_agents:
            x = int(self.rng.randint(0, self.cfg.width))
            y = int(self.rng.randint(0, self.cfg.height))
            if (x,y) not in occ:
                self._agent_pos.append((x,y)); occ.add((x,y))
        self._goals = []
        while len(self._goals) < self.cfg.num_agents:
            x = int(self.rng.randint(0, self.cfg.width))
            y = int(self.rng.randint(0, self.cfg.height))
            if (x,y) not in occ:
                self._goals.append((x,y)); occ.add((x,y))
        obs = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def step(self, action):
        self._t += 1
        actions = np.array(action, dtype=int)
        assert actions.shape == (self.cfg.num_agents,)
        # propose
        proposed = []
        for i,a in enumerate(actions):
            x,y = self._agent_pos[i]
            if   a==1: y = max(0, y-1)
            elif a==2: y = min(self.cfg.height-1, y+1)
            elif a==3: x = max(0, x-1)
            elif a==4: x = min(self.cfg.width-1, x+1)
            proposed.append((x,y))
        # collisions
        counts = {}
        for p in proposed: counts[p] = counts.get(p,0)+1
        new_pos, col_mask = [], [False]*self.cfg.num_agents
        for i,p in enumerate(proposed):
            if counts[p] > 1:
                col_mask[i] = True
                new_pos.append(self._agent_pos[i])
            else:
                new_pos.append(p)
        self._agent_pos = new_pos
        self._last_collisions = int(sum(col_mask))

        # rewards
        rewards = np.zeros(self.cfg.num_agents, dtype=np.float32)
        for i in range(self.cfg.num_agents):
            rewards[i] -= self.cfg.step_penalty
            if col_mask[i]:
                rewards[i] -= self.cfg.collision_penalty

        # goals
        occ = set(self._agent_pos)
        for i in range(self.cfg.num_agents):
            if self._agent_pos[i] == self._goals[i]:
                rewards[i] += self.cfg.goal_reward
                taken = occ.union(set(self._goals))
                while True:
                    gx = int(self.rng.randint(0, self.cfg.width))
                    gy = int(self.rng.randint(0, self.cfg.height))
                    if (gx,gy) not in taken:
                        self._goals[i] = (gx,gy); break

        done = self._t >= self.cfg.max_steps
        obs = self._get_obs()
        info = {"collisions": self._last_collisions, "per_agent_rewards": rewards.copy()}
        if self.render_mode == "human":
            self._render_frame()
        return obs, rewards.sum().astype(np.float32), done, False, info

    def _get_obs(self):
        W,H = max(1,self.cfg.width-1), max(1,self.cfg.height-1)
        out = []
        for (ax,ay) in self._agent_pos: out += [ax/W, ay/H]
        for (gx,gy) in self._goals:     out += [gx/W, gy/H]
        return np.array(out, dtype=np.float32)

    def close(self):
        if self._viewer is not None:
            pygame.display.quit()
            pygame.quit()
            self._viewer = None
            self._surface = None
            self._clock = None
            self._font = None
            self._drone_sprites = None
            self._building_sprite = None

    # ----- rendering helpers -----
    def _try_load(self, path):
        try:
            return pygame.image.load(path).convert_alpha()
        except Exception:
            return None

    def _ensure_window(self, wpx, hpx):
        if self._viewer is None:
            pygame.init()
            self._viewer = pygame.display.set_mode((wpx, hpx))
            pygame.display.set_caption("MultiAgentGridEnv")
            self._surface = pygame.Surface((wpx, hpx), flags=pygame.SRCALPHA).convert_alpha()
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("consolas", 18)

    def _ensure_sprites(self, cell):
        if self._drone_sprites is not None and self._building_sprite is not None:
            return
        pad = 10; size = max(8, cell - pad)
        drone = self._try_load(DRONE_PATH)
        building = self._try_load(BUILDING_PATH)
        if drone is not None:
            drone = pygame.transform.smoothscale(drone, (size, size))
        if building is not None:
            building = pygame.transform.smoothscale(building, (size, size))
        self._drone_sprites = [drone for _ in range(self.cfg.num_agents)]
        self._building_sprite = building

    def _draw_watermark(self, surf, wpx: int, hpx: int):
        if not self._watermark_text:
            return
        text = self._font.render(self._watermark_text, True, (240, 240, 245))
        tw, th = text.get_size()
        pad = 8
        bg = pygame.Surface((tw + 2*pad, th + 2*pad), pygame.SRCALPHA)
        pygame.draw.rect(bg, (0,0,0,120), bg.get_rect(), border_radius=10)
        bg_pos = (wpx - bg.get_width() - 10, 10)
        surf.blit(bg, bg_pos)
        surf.blit(text, (bg_pos[0] + pad, bg_pos[1] + pad))

    def _render_frame(self, return_array: bool = False):
        cell, margin = 64, 2
        wpx = self.cfg.width*cell + (self.cfg.width+1)*margin
        hpx = self.cfg.height*cell + (self.cfg.height+1)*margin + 40
        self._ensure_window(wpx, hpx)
        self._ensure_sprites(cell)

        import pygame as _pg
        for event in _pg.event.get():
            if event.type == _pg.QUIT:
                self.close()
                return

        surf = self._surface
        surf.fill((25,25,30,255))

        # grid
        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                rx = x*cell + (x+1)*margin
                ry = y*cell + (y+1)*margin
                _pg.draw.rect(surf, (40,45,55), (rx,ry,cell,cell), border_radius=8)

        agent_colors = [(66,135,245), (80,200,120), (255,99,132), (255,165,0), (160,95,245)]
        goal_colors  = [(120,170,255), (120,230,170), (255,160,180), (255,205,120), (200,160,255)]

        # goals
        for i,(gx,gy) in enumerate(self._goals):
            rx = gx*cell + (gx+1)*margin
            ry = gy*cell + (gy+1)*margin
            cx,cy = rx+cell//2, ry+cell//2
            if self._building_sprite is not None:
                rect = self._building_sprite.get_rect(center=(cx,cy))
                surf.blit(self._building_sprite, rect)
            else:
                pts = [(cx, ry+8), (rx+cell-8, cy), (cx, ry+cell-8), (rx+8, cy)]
                _pg.draw.polygon(surf, goal_colors[i%len(goal_colors)], pts)

        # agents
        for i,(ax,ay) in enumerate(self._agent_pos):
            rx = ax*cell + (ax+1)*margin
            ry = ay*cell + (ay+1)*margin
            cx,cy = rx+cell//2, ry+cell//2
            drone = self._drone_sprites[i] if self._drone_sprites else None
            if drone is not None:
                rect = drone.get_rect(center=(cx,cy))
                surf.blit(drone, rect)
            else:
                _pg.draw.circle(surf, agent_colors[i%len(agent_colors)], (cx,cy), cell//3)
                idx = self._font.render(str(i+1), True, (15,15,18))
                surf.blit(idx, idx.get_rect(center=(cx,cy)))

        # HUD
        hud = self._font.render(f"t={self._t}  collisions={self._last_collisions}", True, (230,230,235))
        surf.blit(hud, (10, hpx-32))

        # WATERMARK (top-right)
        self._draw_watermark(surf, wpx, hpx)

        # blit + present
        if self._viewer is not None:
            self._viewer.blit(surf, (0,0))
            import pygame as _pg
            _pg.display.flip()

        # record
        if self._recorder is not None:
            self._recorder.append(surf)

        self._clock.tick(self.metadata["render_fps"])

# --------------- Public wrapper class (Gym entry-point) ---------------
class FederatedSingleEnv(MultiAgentGridEnv):
    """Thin wrapper that lets you configure env via gym.make kwargs.

    Examples:
        gym.make("FederatedSingle-v0", grid_size=10, num_agents=4)
        gym.make("FederatedSingle-v0", width=12, height=8, num_agents=5, max_steps=400, seed=123)
    """
    metadata = {"render_modes": ["human","rgb_array"], "render_fps": 8}

    def __init__(
        self,
        render_mode: Optional[str] = "human",
        grid_size: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_agents: int = 5,
        max_steps: int = 160,
        step_penalty: float = 0.01,
        goal_reward: float = 1.0,
        collision_penalty: float = 0.05,
        seed: Optional[int] = 42,
        watermark_text: Optional[str] = "sreejeetm1729",
        record_to: Optional[str] = None,
    ):
        if grid_size is not None:
            width = height = int(grid_size)
        if width is None:  width = 7
        if height is None: height = 7
        assert width >= 3 and height >= 3 and num_agents >= 1

        cfg = GridConfig(
            width=int(width),
            height=int(height),
            num_agents=int(num_agents),
            max_steps=int(max_steps),
            step_penalty=float(step_penalty),
            goal_reward=float(goal_reward),
            collision_penalty=float(collision_penalty),
            seed=seed,
        )

        rec = None
        if record_to is not None:
            if imageio is None:
                raise RuntimeError("Recording requested but imageio/imageio-ffmpeg not installed.")
            rec = PygameRecorder(record_to, fps=self.metadata["render_fps"])

        super().__init__(cfg=cfg, render_mode=render_mode, watermark_text=watermark_text, recorder=rec)

if __name__ == "__main__":
    # Tiny manual demo (will open a pygame window)
    env = FederatedSingleEnv(render_mode="human", grid_size=8, num_agents=3, max_steps=60)
    obs, info = env.reset()
    import time
    for _ in range(60):
        obs, r, done, trunc, info = env.step(env.action_space.sample())
        time.sleep(0.1)
        if done or trunc:
            break
    env.close()
