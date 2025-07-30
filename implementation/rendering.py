import pygame
import numpy as np
import cv2
from typing import Optional, Dict, List

class Renderer:
    def __init__(self, width=1400, height=1000):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.SCALED)
        pygame.display.set_caption("AI Trading Dashboard")
        
        # Font system
        self.font_xlarge = pygame.font.SysFont('Arial', 32, bold=True)
        self.font_large = pygame.font.SysFont('Arial', 24, bold=True)
        self.font_medium = pygame.font.SysFont('Consolas', 20)
        self.font_small = pygame.font.SysFont('Consolas', 16)
        
        # Color scheme
        self.colors = {
            'background': (18, 18, 18),
            'panel': (30, 30, 30),
            'border': (80, 80, 80),
            'text': (240, 240, 240),
            'bull': (0, 200, 100),
            'bear': (220, 50, 50),
            'highlight': (255, 200, 0),
            'chart': (100, 150, 255)
        }
        
        # Layout
        self.padding = {'x': 30, 'y': 30, 'line': 35}
        self.clock = pygame.time.Clock()
        self.frame_count = 0

    def render(self, env):
        """Main rendering function"""
        self.screen.fill(self.colors['background'])
        self.frame_count += 1
        
        # Portfolio Panel
        self._draw_panel(30, 30, 400, 200, "PORTFOLIO")
        self._draw_portfolio_metrics(env)
        
        # Assets Panel
        self._draw_panel(450, 30, 900, 200, "ASSETS")
        self._draw_asset_positions(env)
        
        # Chart Panel
        self._draw_panel(30, 250, 1320, 500, "PRICE HISTORY")
        self._draw_price_chart(env)
        
        # Action Highlight
        if hasattr(env, 'last_action'):
            self._draw_action(env.last_action, env.prices)
        
        pygame.display.flip()
        self.clock.tick(30)

    def _draw_portfolio_metrics(self, env):
        metrics = [
            f"Total Value: ${env.portfolio_value:,.2f}",
            f"Cash: ${env.cash:,.2f}",
            f"Return: {(env.portfolio_value/env.initial_cash-1)*100:+.2f}%",
            f"Step: {env.step_count}/{env.max_steps}",
            f"Cumulative Reward: {getattr(env, 'episode_total_reward', 0):+.2f}"
        ]
        for i, text in enumerate(metrics):
            self._draw_text(text, 50, 60 + i*35, self.font_medium)

    def _draw_asset_positions(self, env):
        headers = ["Asset", "Price", "Shares", "Value", "Trend"]
        for i, header in enumerate(headers):
            self._draw_text(header, 460 + i*180, 60, self.font_medium, self.colors['highlight'])
        
        for i in range(env.num_assets):
            y_pos = 90 + i*30
            trend = "↑" if env.prices[i] >= env.moving_avg[i] else "↓"
            trend_color = self.colors['bull'] if trend == "↑" else self.colors['bear']
            
            cells = [
                f"Asset {i+1}",
                f"${env.prices[i]:.2f}",
                f"{env.shares[i]}",
                f"${env.shares[i] * env.prices[i]:,.2f}",
                trend
            ]
            
            for j, cell in enumerate(cells):
                color = trend_color if j == 4 else self.colors['text']
                self._draw_text(cell, 460 + j*180, y_pos, self.font_small, color)

    def _draw_price_chart(self, env):
        x, y, w, h = 50, 280, 640, 440  # Reduced width from 1280 to 640
        if not hasattr(env, 'price_history') or len(env.price_history) < 2:
            return
            
        history = env.price_history[-50:]
        min_p = min(min(p) for p in history)
        max_p = max(max(p) for p in history)
        range_p = max_p - min_p if max_p != min_p else 1
        
        asset_colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255)   # Magenta
        ]
        
        for i in range(env.num_assets):
            points = []
            for j, prices in enumerate(history):
                px = x + (j * w / len(history))
                py = y + h - (prices[i] - min_p) / range_p * h
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(
                    self.screen,
                    asset_colors[i % len(asset_colors)],
                    False,
                    points,
                    2
                )
        
        # Draw legend to the right of the chart
        legend_x, legend_y = x + w + 20, y + 10  # Right of chart (50+640+20=710, 280+10=290)
        for i, color in enumerate(asset_colors[:env.num_assets]):
            # Draw colored rectangle
            pygame.draw.rect(self.screen, color, (legend_x, legend_y + i*25, 15, 15))
            # Draw asset label
            self._draw_text(f"Asset {i+1}", legend_x + 25, legend_y + i*25, self.font_small, self.colors['text'])
        
        # Draw event flag below legend
        if getattr(env, 'event_flag', 0):
            self._draw_text("Market Event!", legend_x, legend_y + env.num_assets*25 + 10, self.font_small, self.colors['bear'])

    def _draw_action(self, action, prices):
        if action == len(prices) * 3:  # No-op
            return
            
        asset_idx = action // 3
        action_type = action % 3
        action_text = ["HOLD", "BUY", "SELL"][action_type]
        bg_color = self.colors['bull'] if action_type == 1 else \
                  self.colors['bear'] if action_type == 2 else \
                  (100, 100, 255)
        
        # Animated background
        alpha = 128 + int(127 * np.sin(self.frame_count * 0.1))
        s = pygame.Surface((250, 40), pygame.SRCALPHA)
        pygame.draw.rect(s, (*bg_color, alpha), (0, 0, 250, 40), border_radius=5)
        pygame.draw.rect(s, (255, 255, 255, alpha), (0, 0, 250, 40), width=2, border_radius=5)
        
        text = f"{action_text} Asset {asset_idx+1}"
        text_surf = self.font_medium.render(text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(125, 20))
        s.blit(text_surf, text_rect)
        
        self.screen.blit(s, (self.screen.get_width() - 270, 10))

    def _draw_panel(self, x, y, w, h, title=""):
        pygame.draw.rect(self.screen, self.colors['panel'], (x, y, w, h), 0, border_radius=8)
        pygame.draw.rect(self.screen, self.colors['border'], (x, y, w, h), 2, border_radius=8)
        if title:
            self._draw_text(title, x + 20, y + 15, self.font_large, self.colors['highlight'])

    def _draw_text(self, text, x, y, font=None, color=None):
        font = font or self.font_small
        color = color or self.colors['text']
        surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))

    def get_frame(self):
        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))

    def close(self):
        pygame.quit()

def save_gif(env, filename="random_agent.gif", max_steps=200, fps=15):
    """Record random agent as GIF"""
    renderer = Renderer()
    frames = []
    obs, _ = env.reset()
    
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, _, done, truncated, _ = env.step(action)
        env.last_action = action
        
        renderer.render(env)
        frame = renderer.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
        
        if done or truncated:
            break
    
    renderer.close()
    if frames:
        import imageio
        imageio.mimsave(filename, frames, fps=fps, loop=0)
        print(f"Saved GIF to {filename}")

def record_agent_performance(
    env,
    model,
    filename="agent_performance.mp4",
    episodes=3,
    max_steps=100,
    fps=30
) -> Optional[List[Dict]]:
    """Record agent performance as MP4"""
    renderer = Renderer()
    frame_size = (1400, 1000)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
    
    episode_data = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_rewards = []
            done = False
            
            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                env.last_action = action
                episode_rewards.append(reward)
                
                renderer.render(env)
                frame = renderer.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)
                
                if done or truncated:
                    break
            
            episode_data.append({
                "episode": episode + 1,
                "total_reward": sum(episode_rewards),
                "final_value": env.portfolio_value
            })
        
        print(f"Saved performance video to {filename}")
        return episode_data
        
    except Exception as e:
        print(f"Recording failed: {str(e)}")
        return None
        
    finally:
        writer.release()
        renderer.close()