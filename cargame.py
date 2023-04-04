import math as maths
import random
import numpy as np
import pygame
import cv2 as cv


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

# GLOBALS
track_img_path = "track.png"
running = False
caravan = []

# define car class
class Car:
    def __init__(self, x, y, image):
        self.start = x,y
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
        self.x += self.speed * maths.cos(self.angle)
        self.y += self.speed * maths.sin(self.angle)
        # wrap-around
        if self.x < -self.width:    self.x = screen_width
        elif self.x > screen_width: self.x = -self.width
        if self.y < -self.height:   self.y = screen_height
        elif self.y > screen_height: self.y = -self.height

        return self
    
    def reset(self):
        print('reset!')
        self.x, self.y = self.start
        self.speed = 0
        self.acceleration = 0.2
        self.max_speed = 10
        self.angle = 0
        self.rotate_speed = 1
        self.max_angle = 45

    def draw(self):
        rotated_image = pygame.transform.rotate(self.image, -self.angle*180/maths.pi)
        screen.blit(rotated_image, (self.x - rotated_image.get_width()/2, self.y - rotated_image.get_height()/2))

        return self
    
    def __repr__(self) -> str:
        return f"Car({self.x},{self.y})"

class Player(Car):
    def __init__(self, x, y, image):
        Car.__init__(self, x, y, image)
    
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
        # handle car rotation
        if keys[pygame.K_RIGHT]:
            self.angle += 0.05
        elif keys[pygame.K_LEFT]:
            self.angle -= 0.05
        
        return super().update()
    
    def draw(self):
        return super().draw()

# define track class
class Track:
    def __init__(self, num_points, point_range, dim):
        self.num_points = num_points
        self.point_range = point_range
        self.points = []
        self.track_points:np.array = []
        self.generate_points()
        self.image = self.reload_image(dim)
        
    def generate_points(self):
        # generate random points
        for i in range(self.num_points):
            x = random.randint(self.point_range[0], self.point_range[1])
            y = random.randint(self.point_range[0], self.point_range[1])
            self.points.append((x, y))
        
    def draw(self):
        # draw track using generated points
        # pygame.draw.lines(screen, white, True, self.points, 10)
        if not self.image:
            self.reload_image((screen_width,screen_height))
        screen.blit(self.image, (0,0 #(screen_width-self.image.get_width())//2,(screen_height-self.image.get_height())//2
                    ))
        # pygame.draw.circle(screen, (0,200,0), (screen_width//2,screen_height//2), screen_height//2)
        # pygame.draw.circle(screen, (255,255,255), (screen_width//2,screen_height//2), screen_height//4)

        # se2 = 70
        # pygame.draw.ellipse(screen, (0,0,0), (0,0,screen_width,screen_height))
        # pygame.draw.ellipse(screen, (255,255,255), (10,10,screen_width-20,screen_height-20))
        # pygame.draw.ellipse(screen, (0,240,0), (se2,se2,screen_width-(se2*2),screen_height-(se2*2)))

    def reload_image(self, dim:tuple, /)-> pygame.Surface:
        assert isinstance(dim,tuple) and len(dim) == 2, "(width, height)"
        global track_img
        img = cv.imread(track_img_path)
        re_sized_img = cv.resize(img, dim)
        self.track_points = cv.Canny(re_sized_img, 125, 175)
        track_img = cv.cvtColor(self.track_points, cv.COLOR_GRAY2RGB)
        return pygame.image.frombuffer(track_img.tobytes(), track_img.shape[1::-1], "RGB")
    
    def detect_collisions(self, caravan:list):
        for car in caravan:
            if(self.has_colided(car)):
                car.reset()
    
    def has_colided(self, car:Car, /)-> bool:
        rect = cv.boxPoints(((car.x,car.y), (car.width,car.height), car.angle*180/maths.pi))
        y_key = lambda x:x[1]
        x_key = lambda x:x[0]
        y_range = maths.floor( min(rect, key=y_key)[1] ), maths.ceil( max(rect, key=y_key)[1] )
        x_range = maths.floor( min(rect, key=x_key)[0] ), maths.ceil( max(rect, key=x_key)[0] )
        for y in range(*y_range):
            for x in range(*x_range):
                if cv.pointPolygonTest(rect, (x, y), True) >= 0:
                    if self.track_points[y,x]:return True
        return False


def main():
    global running
    # load car image
    car_image = pygame.image.load("car.png")

    # create car object
    car = Player(screen_width/2-50, screen_height/2, car_image)
    caravan.append(car)

    # create track object
    track = Track(20, (100, 700), dim=(screen_width,screen_height))

    # game loop
    running = True
    while running:
        # event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    track.reload_image((screen_width, screen_height))

        # update car and track
        car.update()

        #keep the cars on the track
        track.detect_collisions(caravan)

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


if __name__ == '__main__':
    main()