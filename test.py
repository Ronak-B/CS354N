import pygame, sys ,os,neat,pickle
from pygame.locals import *

WIDTH=800
HEIGHT=800
gen=0

class Level():
    def __init__(self):
        self.blocks=[100,300,500,700]
        self.window=[(400,500),(200,400),(400,600),(300,500)]
        self.rects=[pygame.rect.Rect(100,0,20,400),pygame.rect.Rect(100,500,20,300),
            pygame.rect.Rect(300,0,20,200),pygame.rect.Rect(300,400,20,400), pygame.rect.Rect(500,0,20,400), pygame.rect.Rect(500,600,20,200),
             pygame.rect.Rect(700,0,20,300), pygame.rect.Rect(700,500,20,300)]
    
    def draw(self,display):
        for r in self.rects:
            pygame.draw.rect(display,(0,0,0),r)

class Blob():
    def __init__(self):
        self.x=30
        self.y=450
        self.passed=0
        self.last_passed=0
        self.prev_dir=1
    
    def get_rect(self):
        return pygame.rect.Rect(self.x,self.y,20,20)

    def draw(self,display):
        pygame.draw.rect(display,(0,0,255),(self.x,self.y,20,20))
    
    def move(self,dir):
        if dir==1:
            self.x+=5
        if dir==2:
            self.y-=5
        if dir==3:
            self.x-=5
        if dir==4:
            self.y+=5

pygame.init()
display=pygame.display.set_mode((WIDTH,HEIGHT))


level=Level()
blob=Blob()
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config.txt')
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
                         
#use trained model to create neural net
cpath=os.path.relpath('winner.pickle')
file=open(cpath,'rb')
obj=pickle.load(file)

net=neat.nn.FeedForwardNetwork.create(obj,config)

game_over=False
clock=pygame.time.Clock()

while game_over==False:
    clock.tick(30)
    for e in pygame.event.get():
        if e.type==QUIT:
            pygame.quit()
            exit()
                
    passed=0
    for b in level.blocks:
        if blob.x>b:
            passed+=1
    blob.passed=passed
    if passed<4:
        window=level.window[passed]
        window_top=window[0]
        window_bottom=window[1]
    else:
        game_over=True

    output=net.activate(((blob.y-window_bottom),(blob.y-window_top),blob.y,blob.x))

    m=max(output)

    if output[0]==m:
        if blob.prev_dir!=3:
            blob.move(1)
            blob.prev_dir=1
        else :
            blob.move(3)
            blob.prev_dir=3

    elif output[1]==m:
        if blob.prev_dir!=4:
            blob.move(2)
            blob.prev_dir=2
        else:
            blob.move(4)
            blob.prev_dir=4               
                
    elif output[2]==m:
        if blob.prev_dir!=1:
            blob.move(3)
            blob.prev_dir=3 
        else :
            blob.move(1)
            blob.prev_dir=1 
    elif output[3]==m:
        if blob.prev_dir!=2:
            blob.move(4)
            blob.prev_dir=4
        else :
            blob.move(2)
            blob.prev_dir=2

        
    display.fill([255,255,255])
    level.draw(display)
    #print(blob.x,blob.y)          
    blob.draw(display)
    pygame.display.flip()        
