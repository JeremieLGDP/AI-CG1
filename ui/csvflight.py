#!/usr/bin/env python
import csv
from os import name, wait

from ssl import ALERT_DESCRIPTION_INTERNAL_ERROR
from sys import dont_write_bytecode
from cflib import crazyflie
import logging
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie import Commander
logging.basicConfig(level=logging.ERROR)

import cflib.positioning.position_hl_commander as pc

#URIbase = 'radio://0/27/2M/E7E7E7E7'


def flightplan(filename = "fp1.csv"):
    doc = []
    with open(filename, newline='') as csvfile:
        f = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in f :
            r = []
            for char in row :
                i = float(char)
                r.append(i)
            
            doc.append(r)
        print(doc)
        return doc
        
def flight(filename = "fp1.csv", channel = '01'):
    URIbase = 'radio://0/27/2M/E7E7E7E7'
    adress = URIbase + channel
    URI = uri_helper.uri_from_env(default=adress)
    fp = flightplan(filename)
    cflib.crtp.init_drivers() 
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:       
            mc=pc.PositionHlCommander(scf)
            home = mc.get_position
            #print("home: " + home)
            start=time.time()
            end = False
            while((time.time()-start<30) & end != True):
                mc.take_off()
                for row in fp :
                    mc.go_to(home+row)
                end = True
            mc.go_to(home + [0,0,0.5])
            mc.land()
            



        