
# Import Statements
from __future__ import print_function
import glob
import os
import sys
import utlis
import csv
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import cv2
from scipy.spatial import distance as dist  
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import imutils
import time
import dlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
import ibm_db as db2
# import multiprocessing

#Import Carla Packages
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc


try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# Mode Selection Variables  

h=300
w=300
preview=False   # If set to true will display simulation of Lane Detector
joystick=False  # If set to True will enable Console Controller input, If False enables keyboard control
eye_tracker=True

# Cloud Configuration Variables

# Driver ID as stored in cloud database IBM DB2
driver="Deloitte" 
# dsn credential of the IBM DB2 Cloud database
dsn="DATABASE=BLUDB;HOSTNAME=dashdb-txn-sbox-yp-dal09-04.services.dal.bluemix.net;PORT=50000;PROTOCOL=TCPIP;UID=jxm90623;PWD=g59rhr+62bs94jwx;"

# Optimiser Variable ---Donot Change this value

connect=True    
lane_invades=0
conn = db2.connect(dsn,"", "")
#Global Variables for EyeTracker

EYE_AR_THRESH = 0.25 #Eye aspect Ratio
EYE_AR_CONSEC_FRAMES = 20 # No of frames for which Eye has to be closed before detecting drowsiness
COUNTER = 0    # Counter to track the no of frames in which the eyes were found closed consecutively
COUNTER2=0
ALARM_ON = False #Boolean value to determine status of alarm
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #import feature extraction model
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #extract features of left eye
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] #extract features of rigth eye
vs = VideoStream(src=0).start()
# fieldname=["DThrottle","DSteer","Speed","DBrake","Eye_blinks","lane_invades","Dspeed"]
# with open('data.csv','a',newline='') as f:
#                 thewriter=csv.DictWriter(f,fieldnames=fieldname)
#                 thewriter.writeheader()

def estimateGaussian(X):
    mu = np.mean(X, axis=0)
    sigma2 = np.var(X, axis=0)
    return mu, sigma2
    mu, sigma2 = estimateGaussian(X)



#Function to Track Eyes
def track_eyes():
    global COUNTER
    global COUNTER2
    global ALARM_ON 
    frame = vs.read()
    frame = imutils.resize(frame, width=500)  #initialise frame Dimensions to o/p display
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #set color of frame
    output=-1
    rects = detector(gray, 0)

    for rect in rects:                      #loop calcules facial features for every frame
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if eye_tracker:
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)  # draw a outline on the detected lefteye
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1) # draw a outline on the detected righteye

        if ear < EYE_AR_THRESH:    #if aspect ratio of eyes is less than A fixed threshold then increment counter
            COUNTER += 1
            COUNTER2 += 0.1
           

            if COUNTER >= EYE_AR_CONSEC_FRAMES: #if the no of frames for which eyes are closed exceeds a threshold
                if not ALARM_ON:                #turn alarm on
                    ALARM_ON = True
                    t = Thread(target=sound_alarm,args=("alarm.wav",))
                    t.deamon = True
                    t.start()
                    output=1
                if eye_tracker:
                    cv2.putText(frame, "You are Getting Sleepy!!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:         # if a frame is found in which eyes are open reset the counter.
            COUNTER = 0
            ALARM_ON = False
            output=0
        if eye_tracker:
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    if eye_tracker:
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

    return output

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def sound_alarm(path):
    playsound.playsound(path)

def execquery(sql,con=conn):
    stmt = db2.exec_immediate(con,sql)


# Function to Show Output of Lane Detector Camera
def process_img(image,h=h,w=w,show_preview=preview):
    i=np.array(image.raw_data)
    i2=i.reshape((h,w,4))
    # i3=i2[:,:,:3]
    f=lane_detect(i2)
    cv2.imshow("Result", f)
    cv2.waitKey(1)
    return None

#Function TO Detect Lanes

def lane_detect(img):
    frameWidth= w
    frameHeight = h
    intialTracbarVals = (38,60,12,90)
    count=0
    noOfArrayValues =11
    global arrayCurve, arrayCounter
    arrayCounter=0
    arrayCurve = np.zeros([noOfArrayValues])
    myVals=[]
    #utlis.initializeTrackbars(intialTracbarVals)
    
    # imgWarpPoints = img.copy()
    imgFinal = img.copy()
    imgCanny = img.copy()
    try:

    # imgUndis = utlis.undistort(img)
        imgThres,imgCanny,imgColor = utlis.thresholding(img)
        src = utlis.valTrackbars()
        imgWarp = utlis.perspective_warp(imgThres, dst_size=(frameWidth, frameHeight), src=src)
        # imgWarpPoints = utlis.drawPoints(imgWarpPoints, src)
        imgSliding, curves, lanes, ploty = utlis.sliding_window(imgWarp, draw_windows=False)

        curverad =utlis.get_curve(imgFinal, curves[0], curves[1])
        lane_curve = np.mean([curverad[0], curverad[1]])
        imgFinal = utlis.draw_lanes(img, curves[0], curves[1],frameWidth,frameHeight,src=src)
        #currentCurve = lane_curve // 50
        # if  int(np.sum(arrayCurve)) == 0:averageCurve = currentCurve
        # else:
        #     averageCurve = np.sum(arrayCurve) // arrayCurve.shape[0]
        # if abs(averageCurve-currentCurve) >200: arrayCurve[arrayCounter] = averageCurve
        # else :arrayCurve[arrayCounter] = currentCurve
        # arrayCounter +=1
        # if arrayCounter >=noOfArrayValues : arrayCounter=0
        # cv2.putText(imgFinal, str(int(averageCurve)), (frameWidth//2-70, 70), cv2.FONT_HERSHEY_DUPLEX, 1.75, (0, 0, 255), 2, cv2.LINE_AA)
        # imgFinal= utlis.drawLines(imgFinal,lane_curve)
    except :
        imgFinal=img
        cv2.putText(imgFinal, "No Lane Detected", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
    return imgFinal

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class World(object):
    def __init__(self, carla_world, hud, actor_filter, actor_role_name='hero'):
        self.world = carla_world
        self.actor_role_name = actor_role_name
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        #self.collision_sensor = None
        self.lane_invasion_sensor = None
        #self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.sensor = None

    def restart(self):
        global sensor
        global h
        global w
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter("model3"))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = blueprint.get_attribute('color').recommended_values[1]
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
            if preview:
                cam_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
                cam_bp.set_attribute("image_size_x",f"{w}")  #Set width of frame captured by camera
                cam_bp.set_attribute("image_size_y",f"{h}")  #Set Height of frame captured by camera
                cam_bp.set_attribute("fov","100")
                spawn_point = carla.Transform(carla.Location(x=2.5,z=2.5))
                sensor = self.world.spawn_actor(cam_bp,spawn_point, attach_to=self.player)

        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.lane_invasion_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot,joystick):
        self._autopilot_enabled = start_in_autopilot
        self.joystick=joystick
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                

                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                if not joystick:
                    self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                else:
                    self.joycontrol(self.joystick)
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

   #Configured for PS3 Dual Shock3
    def joycontrol(self,joystick):

        y=joystick.get_axis(2)  # Throttle and Brake are controlled be Axis 2 of Controller (L2 & R2)
        x=joystick.get_axis(0)  # Left & Right steer is controlled by Axis 0 (Left Joystick)
        
        #Steering Sensitivity Fine Tunning
        if (x< 0.3 and x>0) or (x>(-1*0.3) and x<0):  
            x=x/10
        elif (x< 0.5 and x>=0.3) or (x>(-1*0.5) and x<=0.3):
            x=x/5
        elif (x< 0.7 and x>=0.5) or (x>(-1*0.7) and x<=0.5):
            x=x/2
        elif (x< 0.9 and x>=0.7) or (x>(-1*0.9) and x<=0.7):
            x=x/1.5
        else:
            x=x
        self._control.steer=round(x,4)
        if y>=0:                        # If L2(Accelerator) is prssed more than R2(Brake)
            self._control.throttle = y
            self._control.brake= 0
        else:
            self._control.brake= y*(-1)
            self._control.throttle = 0
        

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        # self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.sec_prev=0
        self.sec_prev2=0
        self.tprev=0
        self.sprev=0
        self.bprev=0
        self.speedprev=0

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def reset_invades(self,sec_prev2):
        global lane_invades
        global COUNTER2
        seconds=round(self.simulation_time,1)
        lapse=seconds-sec_prev2
        if lapse > 10:
            lane_invades=0
            COUNTER2 =0
            return seconds
        else:
            return sec_prev2
    
    def write_csv(self,c,v,sec_prev,tprev,sprev,bprev,speedprev):
        global lane_invades
        global connect
        global COUNTER2
        global driver
        seconds=round(self.simulation_time,1)
        lapse=seconds-sec_prev
        dt=c.throttle-tprev
        ds=c.steer-sprev
        db=c.brake-bprev
        dspeed=3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2)-speedprev
        track_eyes()
        
        if lapse > 0.5:
            t=np.array([round(dt,1), round(ds,1), round(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2),0),round(db,1),COUNTER2,math.ceil(lane_invades),dspeed ])
            x=pd.read_csv("data.csv")
            X=x.values
            mu, sigma2 = estimateGaussian(X)
            distribution = multivariate_normal(mean=mu, cov=np.diag(sigma2),allow_singular=True )
            p = distribution.pdf(t) 
            if p<1.38e-18:
                self.notification("Anomaly Detected")
                
                s1=str(round(dt,1))
                s2=str(round(ds,1))
                s4=str(round(db,1))
                s3=str(round(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2),0))
                s5=str(round(COUNTER2,1))
                s6=str(math.ceil(lane_invades))
                sql= "insert into ANOMALY values ("+ s1+ "," + s2+ "," + s3+  ","+s4+ "," +s5+ "," +s6+ "," +str(dspeed)+ ",'" + driver+  "')"
                T=Thread(target=execquery,args=[sql],daemon=True)
                T.start()

            return (seconds,c.throttle,c.steer,c.brake,(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2)))
        else:
            return sec_prev,tprev,sprev,bprev,speedprev
            
        
    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        v = world.player.get_velocity()
        c = world.player.get_control()
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
            self.sec_prev,self.tprev,self.sprev,self.bprev,self.speedprev=self.write_csv(c,v,self.sec_prev,self.tprev,self.sprev,self.bprev,self.speedprev)
            self.sec_prev2=self.reset_invades(self.sec_prev2)
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        # self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        global lane_invades
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        lane_invades+=0.25


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB']
           ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
            if preview==True:
            	sensor.listen(lambda data: process_img(data))

        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


           
def game_loop(args,connect):
    global joystick
    pygame.init()
    pygame.font.init()
    world = None
    if connect==True:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        connect=False
    try:
        if joystick:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
        else:
            joystick=None
        

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args.filter, args.rolename)
        controller = KeyboardControl(world, args.autopilot,joystick)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()

#Main Function

def main():
    
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='600x400',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:

        game_loop(args,connect)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
