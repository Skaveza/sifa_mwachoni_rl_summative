import pygame
import numpy as np
import imageio

class Renderer:
    def __init__(self, width=1000, height=800):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.font = pygame.font.SysFont('Arial', 20)
        self.width = width
        self.height = height
        self.asset_colors = [
            (255, 0, 0), (0, 0, 255), (0, 128, 0),
            (255, 165, 0), (128, 0, 128)
        ]
        
    def render(self, env):
        self.screen.fill((240, 240, 240))
        
        # Portfolio info
        texts = [
            f"Portfolio Value: ${env.portfolio_value:.2f}",
            f"Cash: ${env.cash:.2f}",
            f"Step: {env.step_count}/{env.max_steps}",
            "KYC Compliant" if env.compliance_flag else "KYC Alert"
        ]
        
        for i, text in enumerate(texts):
            surf = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(surf, (10, 10 + i*25))
        
        # Asset info
        for i in range(env.num_assets):
            y_base = 150 + i*120
            pygame.draw.rect(
                self.screen, 
                self.asset_colors[i],
                (10, y_base, 150, 30)
            )
            
            texts = [
                f"Asset {i+1}: ${env.prices[i]:.2f}",
                f"Shares: {env.shares[i]}",
                f"MA: {env.moving_avg[i]:.2f}"
            ]
            
            if env.dividends[i] > 0:
                texts.append(f"Dividend: ${env.dividends[i]:.2f}")
                
            for j, text in enumerate(texts):
                surf = self.font.render(text, True, (0, 0, 0))
                self.screen.blit(surf, (10, y_base + 40 + j*25))
                
            # Price history visualization
            if hasattr(env, 'price_history'):
                hist = [p[i] for p in env.price_history[-50:]]
                if len(hist) > 1:
                    points = []
                    max_p = max(hist)
                    min_p = min(hist)
                    range_p = max_p - min_p if max_p != min_p else 1
                    
                    for x, p in enumerate(hist):
                        x_pos = 200 + x * 5
                        y_pos = y_base + 30 - ((p - min_p)/range_p) * 50
                        points.append((x_pos, y_pos))
                    
                    if len(points) > 1:
                        pygame.draw.lines(self.screen, self.asset_colors[i], False, points, 2)
        
        pygame.display.flip()
    
    def get_frame(self):
        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))
    
    def close(self):
        pygame.quit()

# Standalone function (not part of Renderer class)
def save_gif(env, filename='trading_env.gif', max_steps=100, fps=10):
    """
    Record a GIF of the environment with random actions
    """
    renderer = Renderer()
    frames = []
    state, _ = env.reset()
    
    for _ in range(max_steps):
        action = env.action_space.sample()  # Random actions
        state, reward, done, truncated, _ = env.step(action)
        renderer.render(env)
        frames.append(renderer.get_frame())
        
        if done or truncated:
            break
    
    renderer.close()
    
    # Convert frames to GIF
    imageio.mimsave(filename, frames, fps=fps, loop=0)
    print(f"Saved GIF to {filename}")