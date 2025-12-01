import pygame
import config

class Button:
    # a callback is like leaving a "future instruction" to say "when this happens, do this"
    def __init__(self, rect, text, callback):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.font = pygame.font.SysFont(None, config.FONT_SIZE)

    def draw(self, screen):
        mouse = pygame.mouse.get_pos()
        color = config.BUTTON_HOVER if self.rect.collidepoint(mouse) else config.BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect)
        txt = self.font.render(self.text, True, config.TEXT_COLOR)
        screen.blit(txt, txt.get_rect(center=self.rect.center))

    def click(self, pos):
        if self.rect.collidepoint(pos):
            self.callback()
