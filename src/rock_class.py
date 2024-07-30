#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy

class Rock:
    def __init__(self, x_position, y_position, z_position, id):
        self.id = id
        self.creation_time = rospy.Time.now()
        self.latest_update = self.creation_time
        self.x_position = x_position
        self.y_position = y_position
        self.z_position = z_position
        self.confirmed_rock = False
        self.time_until_rock_is_confirmed = rospy.get_param(rospy.get_name() + '/time_until_rock_is_confirmed')
        self.ml_confirmed = False

    def get_position(self):
        return (self.x_position,self.y_position,self.z_position)
    
    def get_x(self):
        return self.x_position
    def get_y(self):
        return self.y_position
    def get_z(self):
        return self.z_position
    
    def get_id(self):
        return self.id

    def get_confidence(self):
        return self.confidence
    
    def get_creation_time(self):
        return self.creation_time
    
    def get_age_sec(self):
        return (rospy.Time.now() - self.creation_time).to_sec()
    
    def get_time_since_last_updace_sec(self):
        return (rospy.Time.now() - self.latest_update).to_sec()
    
    def confirm_rock(self):
        self.confirmed_rock = True

    def is_confirmed(self):
        return self.confirmed_rock
    
    def ml_confirm_rock(self):
        self.ml_confirmed = True

    def is_ml_confirmed(self):
        return self.ml_confirmed

    def update_position(self, new_rock):
        self.latest_update = rospy.Time.now()

        if self.is_confirmed():
            return

        self.x_position = (self.x_position + new_rock.get_x())/2
        self.y_position = (self.y_position + new_rock.get_y())/2
        self.z_position = (self.z_position + new_rock.get_z())/2

        if self.get_age_sec() > self.time_until_rock_is_confirmed:
            self.confirm_rock()
