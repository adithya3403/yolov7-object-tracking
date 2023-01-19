'''
command: python detect_and_track.py --weights best.pt --source test2.mp4 --vi
link for best.pt: https://drive.google.com/file/d/1HMo-eBzSYNQt5LXb1JiPBxfGFFQBdIZh/view?usp=share_link
'''

import math
import time
import geopy
import os
import cv2
import time
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn
from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal, Command
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
                time_synchronized, TracedModel
from utils.download_weights import download

#For SORT tracking
import skimage
from sort import *

altitude=2.0
lat_long=[]
lat_long.append([28.7041, 77.1025])

#............................... DroneKit Code ...............................
class DroneControl(object):
    def __init__(self, server_enabled=True):
        self.gps_lock = False
        self.altitude = 2.0

        # Connect to the Vehicle
        print('Connected to vehicle.')
        self.vehicle = vehicle
        self.commands = self.vehicle.commands
        self.current_coords = []
        self.webserver_enabled = server_enabled
        print("DroneDelivery Start")

        # Register observers
        # self.vehicle.add_attribute_listener('location', self.location_callback)

    def launch(self):
        print("Waiting for location...")
        while self.vehicle.location.global_frame.lat == 0:
            time.sleep(0.1)
        self.home_coords = [self.vehicle.location.global_frame.lat,
                            self.vehicle.location.global_frame.lon]

        print("Waiting for ability to arm...")
        while not self.vehicle.is_armable:
            time.sleep(.1)

        print('Running initial boot sequence')
        self.change_mode('GUIDED')
        self.arm()
        self.takeoff()

    def takeoff(self):
        print("Taking off")
        self.vehicle.simple_takeoff(self.altitude)
        while self.vehicle.location.global_relative_frame.alt < self.altitude * .95:
            print("Altitude: " + str(self.vehicle.location.global_relative_frame.alt))
            time.sleep(0.5)

    def arm(self, value=True):
        if value:
            print('Waiting for arming...')
            self.vehicle.armed = True
            while not self.vehicle.armed:
                time.sleep(.1)
        else:
            print("Disarming!")
            self.vehicle.armed = False

    def change_mode(self, mode):
        print("Changing to mode: {0}".format(mode))

        self.vehicle.mode = VehicleMode(mode)
        # while self.vehicle.mode.name != mode:
        #     print('  ... polled mode: {0}'.format(mode))
        #     time.sleep(1)

    def return_to_launch(self):
        print("Returning to launch")
        self.vehicle.mode = VehicleMode("RTL")

    def getLat(self):
        return self.vehicle.location.global_relative_frame.lat
    
    def getLon(self):
        return self.vehicle.location.global_relative_frame.lon

    def goto(self, lat, lon):
        alt = self.altitude
        print("Going to: {0}, {1}, {2}".format(lat, lon, alt))
        self.vehicle.simple_goto(LocationGlobalRelative(lat, lon, alt))
    
    def geofence(self, lat, lon):
        # using distance formula in meters using geopy
        distance = geopy.distance.distance((lat, lon), (self.home_coords[0], self.home_coords[1])).m
        if distance > 10:
            print("Geofence breached")
            return False
        else:
            return True

#............................... Bounding Boxes Drawing ............................
"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, names=None, save_with_object_id=False, path=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id) + ":"+ names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)   #centroid of box
        txt_str = ""
        if save_with_object_id:
            txt_str += "%i %i %f %f %f %f %f %f" % (
                id, cat, int(box[0])/img.shape[1], int(box[1])/img.shape[0] , int(box[2])/img.shape[1], int(box[3])/img.shape[0] ,int(box[0] + (box[2] * 0.5))/img.shape[1] ,
                int(box[1] + (
                    box[3]* 0.5))/img.shape[0])
            txt_str += "\n"
            with open(path + '.txt', 'a') as f:
                f.write(txt_str)

        #............................... Display Details ............................
        # plot the center of the screen in green color
        cv2.circle(img, (int(img.shape[1]/2),int(img.shape[0]/2)), 6, (0,255,0),-1)
        # draw a rectangle half the size of the screen in red color
        cv2.rectangle(img, (int(img.shape[1]/4),int(img.shape[0]/4)), (int(img.shape[1]/4*3),int(img.shape[0]/4*3)), (0,0,255), 2)
        # draw x and y axes inside the rectangle in red color
        cv2.line(img, (int(img.shape[1]/4),int(img.shape[0]/2)), (int(img.shape[1]/4*3),int(img.shape[0]/2)), (0,0,255), 2)
        cv2.line(img, (int(img.shape[1]/2),int(img.shape[0]/4)), (int(img.shape[1]/2),int(img.shape[0]/4*3)), (0,0,255), 2)
        # if the whole object is inside the rectangle, draw a green rectangle around it

        locked_flag = False
        if (int(box[0]) > int(img.shape[1]/4) and int(box[1]) > int(img.shape[0]/4) and int(box[2]) < int(img.shape[1]/4*3) and int(box[3]) < int(img.shape[0]/4*3)):
            cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,255,0), 2)
            # display "Locked" on the screen at the top left
            cv2.putText(img, "Locked", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            locked_flag = True

        # else draw a red rectangle around it
        else:
            cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 2)
            # display "Not locked" on the screen at the top left
            cv2.putText(img, "Not locked", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            locked_flag = False
        #..............................................................................
        
        #...................................Angle..............................
        # draw a line from the center of the screen to the center of the object
        cv2.line(img, (int(img.shape[1]/2),int(img.shape[0]/2)), (int((box[0]+box[2])/2),int((box[1]+box[3])/2)), (0,255,0), 2)
        # calculate the angle
        def angle_between(p1, p2):
            xDiff = p2[0] - p1[0]
            yDiff = p2[1] - p1[1]
            angle=math.degrees(math.atan2(yDiff, xDiff))
            if angle<0:
                angle=angle+360
            return angle
        angle = angle_between((int(img.shape[1]/2),int(img.shape[0]/2)), (int((box[0]+box[2])/2),int((box[1]+box[3])/2)))
        # display the angle between east and the line
        cv2.putText(img, str(angle), (int((box[0]+box[2])/2),int((box[1]+box[3])/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 255], 2)
        #..............................................................................
        
        #...................................Latitude and Longitude..............................
        # calculate the latitude and longitude of the object
        init_lat, init_long=28.7041, 77.1025
        lat, long=init_lat+((int((box[0]+box[2])/2)-int(img.shape[1]/2))*altitude/111111), init_long+((int((box[1]+box[3])/2)-int(img.shape[0]/2))*altitude/111111)
        # display the latitude and longitude
        cv2.putText(img, "Lat:"+str(lat), (10, img.shape[0]-30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 0, 0], 2)
        cv2.putText(img, "Long:"+str(long), (10, img.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 0, 0], 2)
        lat_long.append([lat,long])
        #..............................................................................

    return img
#..............................................................................


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id= opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.colored_trk, opt.save_bbox_dim, opt.save_with_object_id
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))


    #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    #......................... 
    
    
    #........Rand Color for every trk.......
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................
   

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()

                txt_str = ""

                #loop over tracks
                for track in tracks:
                    # color = compute_color_for_labels(id)
                    #draw colored tracks
                    if colored_trk:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    rand_color_list[track.id], thickness=2) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                      if i < len(track.centroidarr)-1 ] 
                    #draw same color tracks
                    else:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    (255,0,0), thickness=2) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                      if i < len(track.centroidarr)-1 ] 

                    if save_txt and not save_with_object_id:
                        # Normalize coordinates
                        txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1], track.centroidarr[-1][1] / im0.shape[0])
                        if save_bbox_dim:
                            txt_str += " %f %f" % (np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0], np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
                        txt_str += "\n"
                
                if save_txt and not save_with_object_id:
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(txt_str)

                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names, save_with_object_id, txt_path)
                #........................................................
                
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
        
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                  cv2.destroyAllWindows()
                  raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img or save_with_object_id:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

    # using geopy to calculate distance between two points
    from geopy.distance import geodesic
    init_lat_long=lat_long[0]
    final_lat_long=lat_long[-1]
    distance=geodesic(init_lat_long,final_lat_long).kilometers
    print("The distance travelled by the vehicle is: ", distance, "km")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',help='not download model weights if already exist')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')
    parser.add_argument('--save-bbox-dim', action='store_true', help='save bounding box dimensions with --save-txt tracks')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id to *.txt')
    parser.add_argument('--connect', type=str, default=None, help='dronekit connection string')

    parser.set_defaults(download=True)
    opt = parser.parse_args()
    # print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    # if opt.download and not os.path.exists(str(opt.weights)):
    #     print('Model weights not found. Attempting to download now...')
    #     download('./')

    # if (opt.connect == None):
    #     print("No connection string provided\nConnecting to SITL")
    #     import dronekit_sitl
    #     sitl = dronekit_sitl.start_default()
    #     connection_string = sitl.connection_string()
    # else:
    #     connection_string = opt.connect

    # connecting to the vehicle
    import dronekit_sitl
    sitl=dronekit_sitl.start_default()
    connection_string=sitl.connection_string()
    print('Connecting to vehicle on: %s' % connection_string)
    vehicle = connect(connection_string, wait_ready=False)
    
    drone = DroneControl(vehicle)
    drone.launch()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
