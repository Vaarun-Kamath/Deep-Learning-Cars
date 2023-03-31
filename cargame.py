import pygame
import random
import math


# initialize pygame
pygame.init()

# set up the game screen
screen_width = 1280
screen_height = 720
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Car Race Game")

# set up the clock
clock = pygame.time.Clock()

# define colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

# define car class
class Car:
    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image
        self.width = image.get_width()
        self.height = image.get_height()
        self.speed = 0
        self.acceleration = 0.2
        self.max_speed = 10
        self.angle = 0
        self.rotate_speed = 1
        self.max_angle = 45

    def update(self):
        # update car position
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.speed += self.acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed
        elif keys[pygame.K_DOWN]:
            self.speed -= self.acceleration
            if self.speed < -self.max_speed:
                self.speed = -self.max_speed
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)
        # handle car rotation
        if keys[pygame.K_RIGHT]:
            self.angle += 0.05
        elif keys[pygame.K_LEFT]:
            self.angle -= 0.05
        if self.x < -self.width:
            self.x = screen_width
        elif self.x > screen_width:
            self.x = -self.width
        if self.y < -self.height:
            self.y = screen_height
        elif self.y > screen_height:
            self.y = -self.height


    def draw(self):
        rotated_image = pygame.transform.rotate(self.image, -self.angle*180/math.pi)
        screen.blit(rotated_image, (self.x - rotated_image.get_width()/2, self.y - rotated_image.get_height()/2))

# define track class
class Track:
    def __init__(self, num_points, point_range):
        self.num_points = num_points
        self.point_range = point_range
        self.points = []
        self.generate_points()
        
    def generate_points(self):
        # generate random points
        for i in range(self.num_points):
            x = random.randint(self.point_range[0], self.point_range[1])
            y = random.randint(self.point_range[0], self.point_range[1])
            self.points.append((x, y))
        
    def draw(self):
        # draw track using generated points
        # pygame.draw.lines(screen, white, True, self.points, 10)

        pygame.draw.circle(screen, (0,200,0), (screen_width//2,screen_height//2), screen_height//2)
        pygame.draw.circle(screen, (255,255,255), (screen_width//2,screen_height//2), screen_height//4)
        # se2 = 70
        # pygame.draw.ellipse(screen, (0,0,0), (0,0,screen_width,screen_height))
        # pygame.draw.ellipse(screen, (255,255,255), (10,10,screen_width-20,screen_height-20))
        # pygame.draw.ellipse(screen, (0,240,0), (se2,se2,screen_width-(se2*2),screen_height-(se2*2)))



# load car image
car_image = pygame.image.load("car.png")

# create car object
car = Car(screen_width/2, screen_height/2, car_image)

# create track object
track = Track(20, (100, 700))

# game loop
running = True
while running:
    # event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # update car and track
    car.update()

    # draw background and track
    screen.fill((255, 255, 255))
    track.draw()

    # draw car
    car.draw()

    # update display
    pygame.display.update()

    # set frame rate
    clock.tick(60)

pygame.quit()
