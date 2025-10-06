
from __future__ import annotations
import os, numpy as np
from typing import Optional, Tuple, List

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError("Please install gymnasium: pip install gymnasium") from e

from PIL import Image, ImageDraw

ASSETS_DIR = "assets"
DRONE_PATH = os.path.join(ASSETS_DIR, "drone.png")
BUILDING_PATH = os.path.join(ASSETS_DIR, "building.png")

def _ensure_assets():
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

try:
    import pygame
except Exception:
    pygame = None

# Distinct color themes per environment (board)
# (tile, goal, halo, label)
ENV_THEMES: List[Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]] = [
    ((40, 45, 55),  (210,200,80),  (70,160,255), (200,210,245)),  # blue/gold
    ((40, 52, 48),  (80,210,140),  (255,140,70), (210,245,220)),  # green/orange
    ((52, 44, 52),  (220,120,170), (120,160,255),(245,210,230)),  # pink/purple
    ((50, 46, 40),  (210,170,110), (160,95,245), (245,230,210)),  # tan/violet
    ((42, 42, 54),  (160,210,240), (255,99,132), (220,230,245)),  # indigo/cyan
]

AGENT_COLORS = [(66,135,245),(80,200,120),(255,99,132),(255,165,0),(160,95,245),(255,215,0)]

class FederatedMultiEnv(gym.Env):
    """
    Broadcaster (left) + single-row clients (right) with DISTINCT COLORS per environment.
      - num_envs boards (clients), each themed differently
      - num_agents per board
      - action: (num_agents,) broadcast OR (num_envs, num_agents) per-board
      - render_mode: "rgb_array" or "human"
    Observation: vector (4*num_agents) from board 0.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 grid_size: Optional[int] = None,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 num_agents: int = 2,
                 num_envs: int = 4,
                 max_steps: int = 300,
                 seed: Optional[int] = None):
        super().__init__()
        _ensure_assets()
        if grid_size is not None:
            width = height = int(grid_size)
        self.W = 10 if width  is None else int(width)
        self.H = 10 if height is None else int(height)
        self.A = int(num_agents)
        self.M = int(num_envs)
        assert self.W>=3 and self.H>=3 and self.A>=1 and self.M>=1
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        # spaces
        self.action_space = spaces.MultiDiscrete([5]*self.A)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4*self.A,), dtype=np.float32)

        # state
        self.t = 0
        self.pos  = None             # (M, A, 2)
        self.goal = None             # (M, A, 2)
        self._last_mode = "broadcast"
        self._last_action = None

        # rendering
        self._cell = 64
        self._margin = 2
        self._broad_w = 260
        self._screen = None
        self._surf = None
        self._clock = None
        self._font = None
        self._drone = None
        self._building = None

        if self.render_mode == "human" and pygame is None:
            raise RuntimeError("render_mode='human' requires pygame")

    # ------------- API -------------
    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.pos  = self.rng.integers(0, [self.W, self.H], size=(self.M, self.A, 2))
        self.goal = self.rng.integers(0, [self.W, self.H], size=(self.M, self.A, 2))
        self._last_action = np.zeros((self.A,), dtype=int)
        self._last_mode = "broadcast"
        obs = self._obs_single(0)
        info = {"num_envs": self.M}
        if self.render_mode == "human":
            self._ensure_pygame(); self._render_frame()
        return obs, info

    def step(self, action):
        a = np.asarray(action, dtype=int)
        if a.shape == (self.A,):
            mode = "broadcast"
            act = np.broadcast_to(a, (self.M, self.A))
        elif a.shape == (self.M, self.A):
            mode = "per-board"
            act = a
        else:
            raise AssertionError(f"action must be (num_agents,) or (num_envs, num_agents); got {a.shape}")

        self._last_mode = mode
        self._last_action = a.copy()
        self.t += 1

        for m in range(self.M):
            for i, u in enumerate(act[m]):
                x,y = self.pos[m, i]
                if   u==1: y = max(0, y-1)
                elif u==2: y = min(self.H-1, y+1)
                elif u==3: x = max(0, x-1)
                elif u==4: x = min(self.W-1, x+1)
                self.pos[m, i] = (x,y)

        r = 0.0
        for m in range(self.M):
            for i in range(self.A):
                if (self.pos[m, i] == self.goal[m, i]).all():
                    r += 1.0
                    self.goal[m, i] = self.rng.integers(0, [self.W, self.H], size=(2,))

        obs = self._obs_single(0)
        term = False
        trunc = self.t >= self.max_steps
        info = {"num_envs": self.M, "mode": self._last_mode}
        if self.render_mode == "human":
            self._render_frame()
        return obs, float(r), term, trunc, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame(return_array=True)
        elif self.render_mode == "human":
            self._render_frame()

    def close(self):
        if pygame is not None and self._screen is not None:
            try:
                import pygame as _pg
                _pg.display.quit(); _pg.quit()
            except Exception:
                pass
            self._screen = None

    # ------------- Helpers -------------
    def _obs_single(self, m:int):
        a = self.pos[m].astype(np.float32)  / np.array([max(1,self.W-1), max(1,self.H-1)], dtype=np.float32)
        g = self.goal[m].astype(np.float32) / np.array([max(1,self.W-1), max(1,self.H-1)], dtype=np.float32)
        return np.concatenate([a.ravel(), g.ravel()]).astype(np.float32)

    def _ensure_pygame(self):
        import pygame as _pg
        if self._screen is not None: return
        _pg.init()
        w_board = self.W*self._cell + (self.W+1)*self._margin
        h_board = self.H*self._cell + (self.H+1)*self._margin + 36
        spacing = 16
        W = self._broad_w + spacing + self.M * w_board + 24
        H = h_board + 8
        self._screen = _pg.display.set_mode((W,H))
        _pg.display.set_caption("FederatedMultiEnv (Broadcaster + Distinct Colors)")
        self._surf = _pg.Surface((W,H), flags=_pg.SRCALPHA).convert_alpha()
        self._clock = _pg.time.Clock()
        self._font = _pg.font.SysFont("consolas", 18)
        try:
            self._drone = _pg.transform.smoothscale(_pg.image.load(DRONE_PATH).convert_alpha(), (self._cell-14, self._cell-14))
        except Exception:
            self._drone = None
        try:
            self._building = _pg.transform.smoothscale(_pg.image.load(BUILDING_PATH).convert_alpha(), (self._cell-14, self._cell-14))
        except Exception:
            self._building = None

    def _draw_broadcaster(self, surf, ox, H):
        import pygame as _pg
        _pg.draw.rect(surf, (34,38,48), (ox, 0, self._broad_w, H), border_radius=8)
        title = self._font.render("BROADCASTER", True, (230,230,235))
        surf.blit(title, (ox+12, 8))

        mode = f"mode: {self._last_mode}"
        surf.blit(self._font.render(mode, True, (200,210,220)), (ox+12, 32))
        surf.blit(self._font.render("action (per agent):", True, (180,190,200)), (ox+12, 58))

        bx = ox + 12
        by = 80
        bw = self._broad_w - 24
        cell_h = 28
        for i in range(self.A):
            _pg.draw.rect(surf, (48,54,66), (bx, by + i*cell_h, bw, cell_h-6), border_radius=6)
            label = self._font.render(f"{i+1}", True, (180,180,190))
            surf.blit(label, (bx+6, by + i*cell_h + 2))
            if self._last_action is not None:
                if self._last_action.ndim == 1:
                    val = int(self._last_action[i])
                else:
                    val = int(self._last_action[0, i])
                name = ["stay","up","down","left","right"][val if 0 <= val <= 4 else 0]
                surf.blit(self._font.render(name, True, (220,220,230)), (bx+40, by + i*cell_h + 2))

        # Broadcast arrow
        tmod = (self.t % 30) / 30.0
        arrow_color = (90+int(100*tmod), 180, 255)
        cx = ox + self._broad_w + 8
        y0 = 24
        y1 = H - 24
        _pg.draw.line(surf, arrow_color, (cx, y0), (cx, y1), width=4)
        _pg.draw.polygon(surf, arrow_color, [(cx+12, H//2), (cx, H//2 - 10), (cx, H//2 + 10)])

    def _render_frame(self, return_array: bool=False):
        import pygame as _pg
        if self._screen is None:
            self._ensure_pygame()

        for event in _pg.event.get():
            if event.type == _pg.QUIT:
                return

        w_board = self.W*self._cell + (self.W+1)*self._margin
        h_board = self.H*self._cell + (self.H+1)*self._margin + 36
        spacing = 16
        W = self._broad_w + spacing + self.M * w_board + 24
        H = h_board + 8
        self._surf.fill((25,25,30,255))

        # Broadcaster
        self._draw_broadcaster(self._surf, 8, H-16)

        # Clients
        ox0 = self._broad_w + spacing + 16
        for m in range(self.M):
            theme = ENV_THEMES[m % len(ENV_THEMES)]
            tile_col, goal_col, halo_col, label_col = theme

            ox = ox0 + m * w_board

            # grid tiles (themed)
            for y in range(self.H):
                for x in range(self.W):
                    rx = ox + x*self._cell + (x+1)*self._margin
                    ry =    y*self._cell + (y+1)*self._margin
                    _pg.draw.rect(self._surf, tile_col, (rx,ry,self._cell,self._cell), border_radius=8)

            # goals (tinted squares if no sprite)
            for i in range(self.A):
                gx,gy = self.goal[m, i]
                rx = ox + gx*self._cell + (gx+1)*self._margin
                ry =    gy*self._cell + (gy+1)*self._margin
                cx,cy = rx+self._cell//2, ry+self._cell//2
                if self._building is not None:
                    rect = self._building.get_rect(center=(cx,cy)); self._surf.blit(self._building, rect)
                else:
                    _pg.draw.rect(self._surf, goal_col, (rx+10, ry+10, self._cell-20, self._cell-20), border_radius=6)

            # agents (halo in env color + inner agent color)
            for i in range(self.A):
                ax,ay = self.pos[m, i]
                rx = ox + ax*self._cell + (ax+1)*self._margin
                ry =    ay*self._cell + (ay+1)*self._margin
                cx,cy = rx+self._cell//2, ry+self._cell//2
                # halo
                _pg.draw.circle(self._surf, halo_col, (cx,cy), self._cell//3 + 6)
                # inner
                if self._drone is not None:
                    rect = self._drone.get_rect(center=(cx,cy)); self._surf.blit(self._drone, rect)
                else:
                    _pg.draw.circle(self._surf, AGENT_COLORS[i%len(AGENT_COLORS)], (cx,cy), self._cell//3)

            # board label
            lbl = self._font.render(f"Client {m}", True, label_col)
            self._surf.blit(lbl, (ox+4, H-34))

        # time HUD
        hud = self._font.render(f"t={self.t}", True, (230,230,235))
        self._surf.blit(hud, (10, 6))

        self._screen.blit(self._surf, (0,0)); _pg.display.flip()
        self._clock.tick(self.metadata["render_fps"])

        if return_array:
            arr = _pg.surfarray.array3d(self._screen)
            import numpy as _np
            return _np.transpose(arr, (1,0,2))
        return None
