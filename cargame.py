import math as maths
import random
import numpy as np
import pygame
import cv2 as cv
import helper as hlp
from NeuralNetwork import NN
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
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4
FAIL =0
PI = maths.pi

# define car class
class Car:
    def __init__(self, x, y, image):
        self.start = x,y
        self.x = x
        self.y = y
        self.image = image
        self.width = image.get_width()
        self.height = image.get_height()
        self.speed = 0.5
        self.acceleration = 0.2
        self.max_speed = 10
        self.angle = PI/2
        self.rotate_speed = 1
        self.max_angle = 45
        self.checkpoints = set()


        # self.dist = np.array([0,0,0,0,0]                                #!

    def update(self, direc, /):
        if direc == UP:
            self.speed += self.acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed
        elif direc == DOWN:
            self.speed -= self.acceleration
            if self.speed < -self.max_speed:
                self.speed = -self.max_speed
        
        # handle car rotation
        if direc == RIGHT: self.angle += 0.05
        elif direc == LEFT: self.angle -= 0.05
        if self.angle >PI: self.angle -=2*PI
        if self.angle <-PI: self.angle +=2*PI
        
        # update car position
        self.x += self.speed * maths.cos(self.angle)
        self.y += self.speed * maths.sin(self.angle)
        # wrap-around
        if self.x < -self.width:     self.x = screen_width
        elif self.x > screen_width:  self.x = -self.width
        if self.y < -self.height:    self.y = screen_height
        elif self.y > screen_height: self.y = -self.height
        return self
    
    def reset(self):
        print('reset!')
        self.x, self.y = self.start
        self.speed = 0.5
        self.acceleration = 0.2
        self.max_speed = 10
        self.angle = PI/2
        self.rotate_speed = 1
        self.max_angle = 45
        self.checkpoints = set()

    def draw(self):
        rotated_image = pygame.transform.rotate(self.image, -self.angle*180/maths.pi)
        screen.blit(rotated_image, (self.x - rotated_image.get_width()/2, self.y - rotated_image.get_height()/2))
        # clr = [(255,0,0),(255,255,0),(0,255,0),(0,255,255),(0,0,255)]                                                                       #* DEBUG
        # if len(self.dist) !=5: print(len(self.dist),self.dist)                                                                              #* DEBUG
        # for i,(y,x) in enumerate(self.dist):                                                                                                #* DEBUG
            # pygame.draw.line(screen,clr[i],(self.x,self.y),(self.x+x,self.y+y),1)                                                           #* DEBUG            
        return self
    
    def __repr__(self) -> str:
        return f"Car({self.x},{self.y})"

class Player(Car):
    def __init__(self, x, y, image):
        Car.__init__(self, x, y, image)
    
    def update(self):
        # update car position
        direc = FAIL
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: direc = UP
        elif keys[pygame.K_DOWN]: direc = DOWN
        if keys[pygame.K_RIGHT]: direc = RIGHT
        elif keys[pygame.K_LEFT]: direc = LEFT        
        return super().update(direc)
    
    def draw(self):
        return super().draw()
    
    def __repr__(self) -> str:
        return f"You({self.x},{self.y})"

class Computer(Car):
    def __init__(self, x, y, image):
        self.dist = np.array([0,0,0,0,0]) # W NW N NE E
        # self.dist = self.weights = 0.1*np.random.randn(1,5) # W NW N NE E
        self.dist = self.dist.reshape(len(self.dist),1)
        self.brain = NN()
        self.controls = []
        super().__init__(x, y, image)
    
    def update(self):
        direc = FAIL
        self.compute() # U R D L
        ctrl_max = np.max(self.controls)
        print(f"{len(self.controls)} : ctrl_max = {ctrl_max}")
        if ctrl_max > 0.2 and len(self.controls) == 4:
            direc = self.controls.find(ctrl_max) + 1
            print(f"Direc : {direc}")
        super().update(direc)
    
    def compute(self):
        # TODO - Get Distance in this line always
        # print("------------")
        # print(f"self.dist {self.dist.shape}:")
        # print(self.dist)
        # print("------------")
        self.brain.L1.forward_activate(self.dist)
        self.brain.L2.forward_activate(self.brain.L1.output)
        self.brain.L3.forward_activate(self.brain.L2.output)
        self.controls = self.brain.L3.output.reshape(1,len(self.brain.L3.output))[0]
        # self.controls = self.brain.L3.output
        print("------------")
        print(f"self.brain.L1.output {self.brain.L1.output.shape}:")
        print(self.brain.L1.output)
        print(f"self.brain.L2.output {self.brain.L2.output.shape}:")
        print(self.brain.L2.output)
        print(f"self.brain.L3.output {self.brain.L3.output.shape}:")
        print(self.brain.L3.output)
        print(f"self.controls {self.controls.shape}:")
        print(self.controls)
        print("------------")
    
    def __repr__(self) -> str:
        return f"Comp({self.x},{self.y})"

# define track class
class Track:
    def __init__(self, num_points, point_range, dim):
        self.num_points = num_points
        self.point_range = point_range
        self.points = []
        self.track_points:np.array = []
        self.checkpoints:set[tuple[int,int],tuple[int,int]] = set()
        self.generate_points()
        self.image = self.reload_image(dim)
        
    def generate_points(self):
        # generate random points
        for i in range(self.num_points):
            x = random.randint(self.point_range[0], self.point_range[1])
            y = random.randint(self.point_range[0], self.point_range[1])
            self.points.append((x, y))
        
    def draw(self):
        if not self.image: self.reload_image((screen_width,screen_height))
        screen.blit(self.image, (0,0 #(screen_width-self.image.get_width())//2,(screen_height-self.image.get_height())//2
                    ))

    def reload_image(self, dim:tuple, /)-> pygame.Surface:
        self.track_points = hlp.get_new_img(dim,track_img_path)
        self.checkpoints = hlp.get_checkpoints(self.track_points,dim)
        track_img = cv.cvtColor(self.track_points, cv.COLOR_GRAY2RGB)
        # a,b,crnrs, self.checkpoints = hlp.get_checkpoints(self.track_points,dim)            #* DEBUG
        # for y,x in a:                                                                       #* DEBUG
        #     cv.circle(track_img, (x,y),9,(0,255,0),1)                                       #* DEBUG
        # for y,x in b:                                                                       #* DEBUG
        #     clr = random.randint(50, 255), random.randint(50, 255), random.randint(50, 255) #* DEBUG
        #     cv.circle(track_img, (x,y),9,clr,-1)                                            #* DEBUG
        #     cv.circle(track_img, (x,y),150,clr,1)                                           #* DEBUG
        #     cv.rectangle(track_img, (x-140,y-140),(x+140,y+140),clr,1)                      #* DEBUG
        # for (y1,x1),(y2,x2) in self.checkpoints:                                            #* DEBUG
        #     cv.line(track_img,(x1,y1),(x2,y2),(0,255,0),2)                                  #* DEBUG        
        # for y,x in crnrs:# pt= (x,y)                                                        #* DEBUG
        #     cv.circle(track_img, (int(y),int(x)),5,(255,0,0),-1)                            #* DEBUG
        return pygame.image.frombuffer(track_img.tobytes(), track_img.shape[1::-1], "RGB")
    
    def detect_collisions(self, caravan:list[Car])-> None:
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
    
    def get_distance(self, caravan:list[Car])-> None:
        for car in caravan:
            # if not isinstance(car, Computer): continue                        #!
            dist = [] # W NW N NE E
            for ang in (-PI/2,-PI/4,0,PI/4,PI/2):
                theta = car.angle + ang
                if theta<-PI: theta += 2 * PI
                elif theta>PI: theta -= 2 * PI
                dist.append( 
                    hlp.get_dist(
                        car.x,
                        car.y,
                        theta,
                        self.track_points
                    )
                )
            car.dist = np.array(dist)
    
    def handle_checkpoint(self, caravan:list[Car])-> None:
        for car in caravan:
            if (chk_pt := hlp.get_current_chkpt(car,self.checkpoints-car.checkpoints)) is None:continue
            car.checkpoints.add(chk_pt)            
            # print(len(car.checkpoints),'/',len(self.checkpoints))                                           #* DEBUG
            # track_img = cv.cvtColor(self.track_points, cv.COLOR_GRAY2RGB)                                   #* DEBUG
            # for (y1,x1),(y2,x2) in self.checkpoints:                                                        #* DEBUG
            #     cv.line(track_img,(x1,y1),(x2,y2),(0,255,0),2)                                              #* DEBUG
            # for (y1,x1),(y2,x2) in car.checkpoints:                                                         #* DEBUG
            #     cv.line(track_img,(x1,y1),(x2,y2),(255,0,0),2)                                              #* DEBUG
            # self.image =pygame.image.frombuffer(track_img.tobytes(), track_img.shape[1::-1], "RGB")         #* DEBUG


def main():
    global running
    # load car image
    car_image = pygame.image.load("car.png")

    # create car object
    car = Computer(80, screen_height/2, car_image)
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
        track.handle_checkpoint(caravan)
        track.get_distance(caravan)

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