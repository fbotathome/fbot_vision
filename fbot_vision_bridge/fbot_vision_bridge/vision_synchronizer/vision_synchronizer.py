#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import message_filters

from butia_vision_bridge import VisionBridge

class VisionSynchronizer:
    POSSIBLE_SOURCES = ['camera_info', 'image_rgb', 'image_depth', 'points']

    def syncSubscribers(source_topic_dict, callback, queue_size=1, exact_time=False, slop=0.1):
        subscribers = []
        for source in VisionSynchronizer.POSSIBLE_SOURCES:
            if source in source_topic_dict.keys():
                subscribers.append((source, message_filters.Subscriber(source_topic_dict[source], VisionBridge.SOURCES_TYPES[source])))

        ts = None
        if exact_time:
            ts = message_filters.TimeSynchronizer([x[1] for x in subscribers], queue_size=queue_size)
        else:
            print([x[0] for x in subscribers])
            ts = message_filters.ApproximateTimeSynchronizer([x[1] for x in subscribers], queue_size=queue_size, slop=slop)
        ts.registerCallback(callback)

        return dict(subscribers)