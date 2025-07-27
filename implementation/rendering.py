import pygame
import numpy as np
import imageio

class Renderer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.font = pygame.font.SysFont('arial', 20)
        self.width = width
        self.height = height

    def render(self, env):
        self.screen.fill((255, 255, 255))
        portfolio_text = self.font.render(
            f"Portfolio: ${env.portfolio_value:.2f} | Cash: ${env.cash:.2f}",
            True, (0, 0, 0)
        )
        self.screen.blit(portfolio_text, (10, 10))
        kyc_color = (0, 255, 0) if env.compliance_flag == 1 else (255, 0, 0)
        pygame.draw.circle(self.screen, kyc_color, (self.width - 50, 30), 20)
        kyc_text = self.font.render("KYC", True, (0, 0, 0))
        self.screen.blit(kyc_text, (self.width - 70, 20))
        for i in range(env.num_assets):
            y_offset = 80 + i * 100
            price_text = self.font.render(
                f"Asset {i+1}: ${env.prices[i]:.2f} (MA: {env.moving_avg[i]:.2f})",
                True, (0, 0, 255)
            )
            self.screen.blit(price_text, (10, y_offset))
            pygame.draw.rect(self.screen, (0, 128, 0), (10, y_offset + 30, env.shares[i] * 5, 20))
            if env.dividends[i] > 0:
                div_text = self.font.render(f"Div: ${env.dividends[i]:.2f}", True, (255, 165, 0))
                self.screen.blit(div_text, (10, y_offset + 60))
        pygame.display.flip()

    def get_frame(self):
        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))

    def close(self):
        pygame.quit()

def save_gif(env, filename='trading_env_random.gif', max_steps=100, fps=10):
    renderer = Renderer()
    frames = []
    state, _ = env.reset()
    for _ in range(max_steps):
        action = env.action_space.sample()
        state, reward, done, truncated, _ = env.step(action)
        renderer.render(env)
        frames.append(renderer.get_frame())
        if done or truncated:
            break
    renderer.close()
    imageio.mimsave(filename, frames, fps=fps)