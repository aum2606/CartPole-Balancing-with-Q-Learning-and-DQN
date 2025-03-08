import pygame
import numpy as np

class CartPoleVisualizer:
    def __init__(self, width=800, height=400):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CartPole Simulation")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        
        # Cart and pole dimensions
        self.cart_width = 80
        self.cart_height = 40
        self.pole_length = 100
        self.pole_thickness = 10
        
        # Scale factors for visualization
        self.cart_scale = 50  # pixels per meter for cart position
        self.clock = pygame.time.Clock()
        
    def render(self, state):
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Extract state variables
        cart_pos, _, pole_angle, _ = state
        
        # Convert cart position to screen coordinates
        cart_x = self.width // 2 + cart_pos * self.cart_scale
        cart_y = self.height // 2
        
        # Draw cart
        cart_rect = pygame.Rect(
            cart_x - self.cart_width // 2,
            cart_y - self.cart_height // 2,
            self.cart_width,
            self.cart_height
        )
        pygame.draw.rect(self.screen, self.BLACK, cart_rect)
        
        # Calculate pole end position
        pole_x = cart_x + self.pole_length * np.sin(pole_angle)
        pole_y = cart_y - self.pole_length * np.cos(pole_angle)
        
        # Draw pole
        pygame.draw.line(
            self.screen,
            self.RED,
            (cart_x, cart_y),
            (pole_x, pole_y),
            self.pole_thickness
        )
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
        
    def close(self):
        pygame.quit()
        
    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False 