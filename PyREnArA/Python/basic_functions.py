### TITLE: PyREnArA (basic functions) ###

### AUHTORS: 
# Robin John (Institute of Prehistoric Archeology, University of Cologne, Bernhard-Feilchenfeld-Str. 11, 50969 Köln, Germany),
# Florian Linsel (Institute of Computer Science, Martin-Luther-Universität Halle-Wittenberg, Von-Seckendorff-Platz 1, 06120 Halle)



# import required packages/libraries
import cv2
import numpy as np
import math
import pandas as pd
import sys
import os
import warnings
from imutils import perspective
import shutil
import winsound
from imutils import contours
from scipy.spatial import distance as dist



### combined metrics functions ###

def combined_metrics(directory):
    
    valid_imageformats = ['.jpg', '.png', '.tif']
    
    for filename in os.listdir(directory):

        imageformat = os.path.splitext(filename)[1]

        if imageformat.lower() not in valid_imageformats:
            continue

        if filename.endswith('color.tif'):
            continue
            
        else:
            name = set_imagename(filename)

            # create folders
            folder_directory = create_folder(name, directory)
            outline_folder = create_folder(name + "_outlines", folder_directory)

            # import original image
            cnts, img, blank_image = import_img(directory + name + imageformat.lower())

            # rotate image and export
            rotated_outlines_image = rotated_outlines_img(cnts, blank_image, folder_directory + name + '_rotated.tif')

            # reimport rotated image
            rotated_cnts, img_rotated, blank_rotated = load_img(rotated_outlines_image)
            img_rotated_1 = img_rotated.copy()
            img_rotated_2 = img_rotated.copy()

            # export single outlines for GMM
            export_for_gmm = export_single_rotated_outlines(blank_rotated, rotated_cnts, outline_folder + name + '_{}.jpg')

            # prepair image for retouch coloring
            retouch_img_prep(img_rotated, rotated_cnts, folder_directory + name + '_retouch_prep.tif')

            # extract metrics
            ref_Obj = scale(cnts)
            ID, site, layer, width, length, l_w_index, area, percentage_area, contour_length, area_contour_ratio, longest_part_location, widest_part_location, lower_part_length, lower_part_relative, upper_part_length, upper_part_relative, upper_lower_ratio, left_part_width, left_part_relative, right_part_width, right_part_relative, left_right_ratio, width_1, width_2, width_3, width_4, width_5, width_6, width_7, length_1, length_2, length_3, length_4, length_5, tip_angle, midpoint_x_offset, midpoint_y_offset, all_angles = metrics(img_rotated_1, name, rotated_cnts, ref_Obj, folder_directory + name + '_show_metrics.tif')

            # select metrical data of interest and export
            metrics_data = zip(ID, site, layer, width, length, l_w_index, area, percentage_area, contour_length, area_contour_ratio, longest_part_location, widest_part_location, lower_part_length, lower_part_relative, upper_part_length, upper_part_relative, upper_lower_ratio, left_part_width, left_part_relative, right_part_width, right_part_relative, left_right_ratio, tip_angle, midpoint_x_offset, midpoint_y_offset)
            metric_csv = write_as_csv(metrics_data, folder_directory + name + '_data_metrics')

            angle_data = zip(ID, all_angles)
            angle_csv = write_as_csv(angle_data, folder_directory + name + '_data_angles')
            
            # retouch analysis
            #red_img, blue_img, green_img, yellow_img, purpel_img = import_color_img(directory + name + '_color.tif')
            #retouch_data = combined_retouch_analysis(img_rotated_2, name, folder_directory + name + '_show_retouches.tif', red_img, blue_img, green_img, yellow_img, purpel_img, rotated_cnts, ref_Obj)
            #export_retouch_data(retouch_data, folder_directory + name + '_data_retouches.csv')
            
            print('Finished ' + name)

            continue
        
### primary functions ###

# scale function
def scale(cnts):

    ref_Obj = None
    area = 150  # set minimum area for objects to be taken into account

    # iterating through artefact contours
    for cnt in cnts:

        # sort out too small objects
        if cv2.contourArea(cnt) < area:
            continue

        # find refernce object by its shape (four corner points)
        contour_length = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.009 * contour_length, True)
        obj_cor = len(approx)
        if obj_cor == 4:

            # create bounding box
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box

            # calculate midpoint on the left and right side
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # measure distance between the points
            d = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if ref_Obj is None:
                # devide distance in pixel by distance in mm to return quotient
                ref_Obj = d / 10

    return ref_Obj

# metrics function
def metrics(img, name, cnts, ref_Obj, show_img_name):

    # create lists for data export
    ID = ['ID']
    site = ['site']
    layer = ['layer']
    width = ['width']
    length = ['length']
    length_width_index = ['l_w_index']
    width_1 = ['w1']
    width_2 = ['w2']
    width_3 = ['w3']
    width_4 = ['w4']
    width_5 = ['w5']
    width_6 = ['w6']
    width_7 = ['w7']
    length_1 = ['l1']
    length_2 = ['l2']
    length_3 = ['l3']
    length_4 = ['l4']
    length_5 = ['l5']
    area = ['area']
    percentage_area = ['percent_area']
    contour_length = ['contour_length']
    area_contour_ratio = ['area_contour_ratio']
    longest_part_location = ['position_longest_extend']
    widest_part_location = ['position_widest_extend']
    lower_part_length = ['length_lower_part']
    lower_part_relative = ['relative_length_lower_part']
    upper_part_length = ['length_upper_part']
    upper_part_relative = ['relative_length_upper_part']
    upper_lower_ratio = ['upper_to_lower_ratio']
    left_part_width = ['width_left_part']
    left_part_relative = ['relative_width_left_part']
    right_part_width = ['width_right_part']
    right_part_relative = ['relative_width_right_part']
    left_right_ratio = ['left_to_right_ratio']
    midpoint_x_offset = ['MP_CM_x_offset']
    midpoint_y_offset = ['MP_CM_y_offset']
    all_angles = [' 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16']
    tip_angles = ['tip_angle']

    i = 1   # set artefact ID

    # iterate through artefact contours
    for cnt in cnts:
        
        warnings.filterwarnings("ignore")
        
        cnt_length = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * cnt_length, True)

        # sort out reference object by its shape
        obj_cor = len(approx)
        if obj_cor == 4:
            continue

        # extract sitename
        sitename = name.split('_')[0].replace('-', ' ')
        
        # extract layer
        underscore = '_'
        if underscore in name:
            second_split = name.split('_')[1]
            if second_split[0:5] == 'layer':
                layername = second_split.split('-')[1]
            else:
                layername = ''
        else:
            layername = ''
            
        # create bounding box
        x, y, w, h = cv2.boundingRect(approx)
        tl = (x, y)
        bl = (x, y + h)
        br = (x + w, y + h)
        tr = (x + w, y)
        (tltrX, tltrY) = midpoint(tl, tr)

        # calculate artefact area
        artefact_area_computed = cv2.contourArea(cnt)
        box_area = w * h
        pc_area = int((artefact_area_computed / box_area) * 100)
        artefact_area_metrical = int(artefact_area_computed / (ref_Obj * ref_Obj))
        
        # calculate length of artefact contour
        cnt_len = int((cv2.arcLength(cnt, True) / ref_Obj))

        # calculate area to contour length ratio
        area_to_contour = int((artefact_area_metrical / cnt_len) * 100)

        # find widest part and width
        wpoints = widthpoints(tl, tr, h, cnt, w, ref_Obj)   # call widthpoints helperfunction
        left = wpoints[0]
        right = wpoints[1]
        wid = int(wpoints[2])

        # find longest part and length
        lpoints = lengthpoints(tl, bl, h, cnt, w, ref_Obj)  # call lengthpoints helperfunction
        top = lpoints[0]
        bot = lpoints[1]
        leng = int(lpoints[2])

        # calculate l-w-index
        l_w_index = np.round(leng / wid, 1)

        # find checkpoints using checkpoints helperfunction
        cpoints = checkpoints(img, left, tl, bl, right, tr, br, top, bot, cnt, w, h, pc_area, ref_Obj)

        # calculate widths of upper and lower part using part_widths helperfunction
        widths = part_widths(img, cpoints, ref_Obj, pc_area, left)
        w1 = widths[0]
        w2 = widths[1]
        w3 = widths[2]
        w4 = widths[3]
        w5 = widths[4]
        w6 = widths[5]
        w7 = widths[6]

        # calculate lenghts of left and right part using length_zone_measure helperfunction
        lengths = length_zone_measure(img, top, tl, bl, bot, cnt, w, h, ref_Obj, pc_area, cpoints)
        l1 = lengths[0]
        l2 = lengths[1]
        l3 = lengths[2]
        l4 = lengths[3]
        l5 = lengths[4]

        # calculate position of widest and longest extent
        max_len_pos = max_length_pos(top, w, tl)
        max_wid_pos = max_width_pos(left, h, bl)

        # calculate absolute and relative length of upper and lower part
        length_midpoint = int(top[0]), int(left[1])
        low_part_length = dist.euclidean(bot, length_midpoint) / ref_Obj
        low_part_rel = int((low_part_length / leng) * 100)
        upp_part_length = dist.euclidean(top, length_midpoint) / ref_Obj
        upp_part_rel = int((upp_part_length / leng) * 100)

        # calculate absolute and relative width of left and right part
        lef_part_width = dist.euclidean(top, (int(left[0]), int(top[1]))) / ref_Obj
        lef_part_rel = int((lef_part_width / wid) * 100)
        rig_part_width = dist.euclidean(top, (int(right[0]), int(top[1]))) / ref_Obj
        rig_part_rel = int((rig_part_width / wid) * 100)

        # calculate upper to lower length ratio
        upper_to_lower = np.round(upp_part_length / low_part_length, 1)

        # calculate left to right width ratio
        left_to_right = np.round(lef_part_width / rig_part_width, 1)

        # calculate all angles at each checkpoint using the angles helperfunction
        angle = angles(cpoints)
        tip_angle = int(angle[7])

        # calculate vector between MP and CM using the midpoint_offset helperfunction
        x_offset, y_offset = midpoint_offset(img, cnt, tl, br, ref_Obj)

        # style and arrange output image
        (rblX, rblY) = midpoint(right, br)
        (rtlX, rtlY) = midpoint(right, tr)
        (ttlX, ttlY) = midpoint(top, tl)
        (ttrX, ttrY) = midpoint(top, tr)

        cv2.line(img, ((tr[0] + 15), right[1]), ((tr[0] + 5), right[1]), (140, 35, 105), 2)
        cv2.line(img, (top[0], (top[1] - 15)), (top[0], (top[1] - 5)), (140, 35, 105), 2)
        cv2.line(img, left, right, (0, 0, 255), 2)
        cv2.line(img, top, bot, (0, 0, 255), 2)
        cv2.line(img, (int(tr[0] + 10), int(tr[1])), (int(br[0] + 10), int(br[1])), (100, 100, 100), 2)
        cv2.line(img, (int(tl[0]), int(tl[1]) - 10), (int(tr[0]), int(tr[1] - 10)), (100, 100, 100), 2)

        cv2.putText(img, str(max_len_pos), (int(top[0] + 2), int(top[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 35, 105), 2)
        cv2.putText(img, str(max_wid_pos), (right[0] + 35, right[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 35, 105), 2)
        cv2.putText(img, "{:.0f}".format(low_part_length) + '/' + "{:.0f}".format(low_part_rel) + ' %', (int(rblX + 15), int(rblY)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
        cv2.putText(img, "{:.0f}".format(upp_part_length) + '/' + "{:.0f}".format(upp_part_rel) + ' %', (int(rtlX + 15), int(rtlY)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
        cv2.putText(img, "{:.0f}".format(lef_part_width), (int(ttlX - 10), int(ttlY - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
        cv2.putText(img, "{:.0f}".format(rig_part_width), (int(ttrX), int(ttrY - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
        cv2.putText(img, "{:.0f}".format(wid), (left[0] - 35, left[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, str('cnt length: ') + "{:.0f}".format(cnt_len), (int(tltrX) - 25, int(tltrY) - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 150, 200), 2)
        cv2.putText(img, str('area: ') + "{:.0f}".format(artefact_area_metrical) + '/' + "{:.0f}".format(pc_area) + ' %', (int(tltrX) - 25, int(tltrY) - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 150, 200), 2)
        cv2.putText(img, str('area/cnt: ') + str(area_to_contour), (int(tltrX) - 25, int(tltrY) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 150, 200), 2)
        cv2.putText(img, "{:.0f}".format(leng), (int(bot[0]) - 15, int(bot[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, str(i), (int(bot[0] - 8), int(bot[1]) + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img, str('l-w-index: ') + "{:.1f}".format(l_w_index), (right[0] + 35, right[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, "{:.1f}".format(x_offset) + '/' + "{:.1f}".format(y_offset), (int(right[0] + 35), int(right[1] + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
        cv2.putText(img, str('MP CM vector: '), (int(right[0] + 35), int(right[1] + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)

        #cv2.imshow('metrics', img)
        #cv2.waitKey(0)

        # append data for data export
        ID.append(name + '_' + "%02d" % i)
        site.append(sitename)
        layer.append(layername)
        width.append(np.round(wid, 1))
        length.append(np.round(leng, 1))
        length_width_index.append(l_w_index)
        width_1.append(w1)
        width_2.append(w2)
        width_3.append(w3)
        width_4.append(w4)
        width_5.append(w5)
        width_6.append(w6)
        width_7.append(w7)
        length_1.append(l1)
        length_2.append(l2)
        length_3.append(l3)
        length_4.append(l4)
        length_5.append(l5)
        area.append(artefact_area_metrical)
        percentage_area.append(np.round(pc_area, 1))
        contour_length.append(cnt_len)
        area_contour_ratio.append(area_to_contour)
        longest_part_location.append(max_len_pos)
        widest_part_location.append(max_wid_pos)
        lower_part_length.append(np.round(low_part_length, 1))
        upper_part_length.append(np.round(upp_part_length, 1))
        left_part_width.append(np.round(lef_part_width, 1))
        right_part_width.append(np.round(rig_part_width, 1))
        lower_part_relative.append(low_part_rel)
        upper_part_relative.append(upp_part_rel)
        left_part_relative.append(lef_part_rel)
        right_part_relative.append(rig_part_rel)
        upper_lower_ratio.append(upper_to_lower)
        left_right_ratio.append(left_to_right)
        midpoint_x_offset.append(x_offset)
        midpoint_y_offset.append(y_offset)
        all_angles.append(angle)
        tip_angles.append(tip_angle)

        # increase ID for next artefact
        i += 1

        # write output image showing metrical data to folder directory
        cv2.imwrite(show_img_name, img)

    return ID, site, layer, width, length, length_width_index, area, percentage_area, contour_length, area_contour_ratio, longest_part_location, widest_part_location, lower_part_length, lower_part_relative, upper_part_length, upper_part_relative, upper_lower_ratio, left_part_width, left_part_relative, right_part_width, right_part_relative, left_right_ratio, width_1, width_2, width_3, width_4, width_5, width_6, width_7, length_1, length_2, length_3, length_4, length_5, tip_angles, midpoint_x_offset, midpoint_y_offset, all_angles

# combined retouch function
def combined_retouch_analysis(img, name, output_name, red_img, blue_img, green_img, yellow_img, purpel_img, rotated_cnts, ref_Obj):

    # call retouch_detect function for each color/retouch style
    dorsal_ID = retouch_analysis(img, name, red_img, rotated_cnts, [0, 0, 255], 'dorsal', ref_Obj)
    backing_ID = retouch_analysis(img, name, blue_img, rotated_cnts, [255, 0, 0], 'backing', ref_Obj)
    ventral_ID = retouch_analysis(img, name, green_img, rotated_cnts, [0, 255, 0], 'ventral', ref_Obj)
    bifacial_ID = retouch_analysis(img, name, yellow_img, rotated_cnts, [0, 255, 255], 'bifacial', ref_Obj)
    surface_ID = retouch_analysis(img, name, purpel_img, rotated_cnts, [255, 0, 255], 'surface', ref_Obj)

    # combine data obtained by retouch_detect by merging them together
    retouch_info = dorsal_ID + backing_ID + ventral_ID + bifacial_ID + surface_ID

    # write retouch image to folder directory
    cv2.imwrite(output_name, img)

    return retouch_info

# find retouches on artefact and measure them
def retouch_analysis(img, name, masked_img, rotated_cnts, color, style, ref_Obj):

    retouch_id = []     # create list for retouch ID

    i = 1   # set ID

    # iterate through artefact contours
    for rotated_cnt in rotated_cnts:

        contour_length = cv2.arcLength(rotated_cnt, True)
        approx = cv2.approxPolyDP(rotated_cnt, 0.01 * contour_length, True)

        # sort out reference object by its shape
        obj_cor = len(approx)
        if obj_cor == 4:
            continue

        # create bounding box
        x, y, w, h = cv2.boundingRect(approx)
        tl = x, y
        tr = x + w, y
        bl = (x, y + h)

        # enlarge box to obtain color-searching area around artefact
        colorbox_tl = int(tl[0] - 17), int(tl[1])
        colorbox_tr = int(tr[0] + 17), int(tr[1])

        # extend color-searching area to search for surface-retouch
        surface_colorbox_tl = int(tl[0] - 23), int(tl[1])
        surface_colorbox_tr = int(tr[0] + 23), int(tr[1])

        # find extreme points to distinguish between left and right lateral
        topmost = tuple(rotated_cnt[rotated_cnt[:, :, 1].argmin()][0])
        botmost = tuple(rotated_cnt[rotated_cnt[:, :, 1].argmax()][0])

        # define point at mid of height to estimate between retouch position
        (tlblX, tlblY) = midpoint(tl, bl)
        mid_lat = int(tlblX), int(tlblY)

        # set lists for lateral points on contour
        lateral_pts_left = []
        lateral_pts_right = []

        # find all lateral points on contour
        ycount = 0
        for ypts in range(h):
            ycount += 1
            ylef_pts = int(tl[0] - 10), int(tl[1] + ycount)
            xlef_pts = cnt_point_left(ylef_pts, rotated_cnt, w + 10)
            if xlef_pts != None:
                lateral_pts_left.append(xlef_pts)
            yrig_pts = int(tr[0] + 10), int(tr[1] + ycount)
            xrig_pts = cnt_point_right(yrig_pts, rotated_cnt, w + 10)
            if xrig_pts != None:
                lateral_pts_right.append(xrig_pts)

        # calculate length of left and right side
        lat_pts_left = np.array(lateral_pts_left, dtype=np.int32)
        left_lateral_length = cv2.arcLength(lat_pts_left, False) / ref_Obj

        lat_pts_right = np.array(lateral_pts_right, dtype=np.int32)
        right_lateral_length = cv2.arcLength(lat_pts_right, False) / ref_Obj

        # import all colored points from colormask_points helperfunction
        color_points_total = colormask_points(masked_img)

        # iterate through colored points
        for color_points in color_points_total:

            lat_left = []
            lat_right = []

            dorsal = []
            ventral = []

            for cpoint in color_points:

                # check if the point is within the color-searching area
                if colorbox_tl[0] < int(cpoint[0]) < colorbox_tr[0]:

                    # check if the point is left or right of the turning point (topmost)
                    # and if so, append it to lists
                    if int(cpoint[0]) < int(topmost[0]):
                        xleft = cnt_point_left(cpoint, rotated_cnt, (w + 20))
                        if xleft == None:
                            continue
                        cv2.circle(img, xleft, 2, color, -1)
                        lat_left.append(xleft)

                    if int(topmost[0]) < int(cpoint[0]):
                        xright = cnt_point_right(cpoint, rotated_cnt, (w + 20))
                        if xright == None:
                            continue
                        cv2.circle(img, xright, 2, color, -1)
                        lat_right.append(xright)

                # check if the point is within the enlarged color-searching area for surface retouch
                elif surface_colorbox_tl[0] < int(cpoint[0]) < surface_colorbox_tr[0]:

                    # check if the point is left or right of the turning point (topmost)
                    # and if so, append it to lists
                    if int(cpoint[0]) < int(topmost[0]):
                        xleft = cnt_point_left(cpoint, rotated_cnt, (w + 25))
                        xright = cnt_point_right((cpoint[0] + (w + 25), cpoint[1]), rotated_cnt, (w + 25))
                        if xleft == None:
                            continue
                        cv2.circle(img, xleft, 2, color, -1)
                        dorsal.append(xleft)
                        cv2.circle(img, xright, 2, color, -1)
                        dorsal.append(xright)

                    if int(topmost[0]) < int(cpoint[0]):
                        xright = cnt_point_right(cpoint, rotated_cnt, (w + 25))
                        if xright == None:
                            continue
                        ventral.append(xright)

            # take the "left" list if they aren´t empty and calculate absolute and relative length
            if len(lat_left) != 0:
                retouch_left = np.array(lat_left, dtype=np.int32)
                ret_len_left = cv2.arcLength(retouch_left, False) / ref_Obj
                rel_ret_len_lef = ret_len_left / left_lateral_length

                # give retouch positional argument
                if retouch_left[1][1] <= int(topmost[1] + 5) and retouch_left[-1][1] >= int(botmost[1] - 5):
                    ret_pos_lef = 'lateral'
                elif retouch_left[1][1] <= int(topmost[1] + 5) and retouch_left[-1][1] >= mid_lat[1]:
                    ret_pos_lef = 'tip_and_medial'
                elif retouch_left[1][1] <= int(topmost[1] + 5):
                    ret_pos_lef = 'tip'
                elif retouch_left[-1][1] >= int(botmost[1] - 5) and retouch_left[1][1] <= mid_lat[1]:
                    ret_pos_lef = 'base_and_medial'
                elif retouch_left[-1][1] >= int(botmost[1] - 5):
                    ret_pos_lef = 'base'
                elif retouch_left[1][1] > int(topmost[1] + 5) and retouch_left[-1][1] < int(botmost[1] - 5):
                    ret_pos_lef = 'medial'

                # create retouch ID with site-name, artefact number, retouch-style, side of retouch, position of
                # retouch and absolute and relative length of retouch
                retouch_id.append(name + '_' + "%02d" % i + ',' + style + ',' + 'left' + ',' + str(ret_pos_lef) + ',' + str(np.round(ret_len_left, 2)) + ',' + str(np.round(rel_ret_len_lef, 2)))

                # style output image
                cv2.putText(img, str('left ') + ret_pos_lef, (tl[0] - 80, retouch_left[1][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                cv2.putText(img, str(style), (tl[0] - 80, retouch_left[1][1] - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                cv2.putText(img, "{:.1f}".format(np.round(ret_len_left, 2)), (tl[0] - 80, retouch_left[1][1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                cv2.putText(img, "{:.1f}".format(rel_ret_len_lef), (tl[0] - 80, retouch_left[1][1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

            # take the "right" list if they aren´t empty and calculate absolute and relative length
            if len(lat_right) != 0:
                retouch_right = np.array(lat_right, dtype=np.int32)
                ret_len_right = cv2.arcLength(retouch_right, False) / ref_Obj
                rel_ret_len_rig = ret_len_right / right_lateral_length

                # give retouch positional argument
                if retouch_right[1][1] <= int(topmost[1] + 5) and retouch_right[-1][1] >= int(botmost[1] - 5):
                    ret_pos_rig = 'lateral'
                elif retouch_right[1][1] <= int(topmost[1] + 5) and retouch_right[-1][1] >= mid_lat[1]:
                    ret_pos_rig = 'tip_and_medial'
                elif retouch_right[1][1] <= int(topmost[1] + 5):
                    ret_pos_rig = 'tip'
                elif retouch_right[-1][1] >= int(botmost[1] - 5) and retouch_right[1][1] <= mid_lat[1]:
                    ret_pos_rig = 'base_and_medial'
                elif retouch_right[-1][1] >= int(botmost[1] - 5):
                    ret_pos_rig = 'base'
                elif retouch_right[1][1] > int(topmost[1] + 5) and retouch_right[-1][1] < int(botmost[1] - 5):
                    ret_pos_rig = 'medial'

                # create retouch ID with site-name, artefact number, retouch-style, side of retouch, position of
                # retouch and absolute and relative length of retouch
                retouch_id.append(name + '_' + "%02d" % i + ',' + style + ',' + 'right' + ',' + str(ret_pos_rig) + ',' + str(np.round(ret_len_right, 2)) + ',' + str(np.round(rel_ret_len_rig, 2)))

                # style output image
                cv2.putText(img, str('right ') + ret_pos_rig, (tr[0] + 10, retouch_right[1][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                cv2.putText(img, str(style), (tr[0] + 10, retouch_right[1][1] - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                cv2.putText(img, "{:.1f}".format(np.round(ret_len_right, 2)), (tr[0] + 10, retouch_right[1][1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                cv2.putText(img, "{:.1f}".format(rel_ret_len_rig), (tr[0] + 10, retouch_right[1][1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

            # take the "dorsal" list if they aren´t empty and calculate absolute and relative area
            if len(dorsal) != 0:
                dorsal_surface_retouch_outline = np.array(dorsal, dtype=np.int32)
                #sorted_dorsal_surface_retouch_outline = sorted(dorsal_surface_retouch_outline, key=lambda k: [k[1], k[0]])
                cv2.drawContours(img, [dorsal_surface_retouch_outline], -1, color, cv2.FILLED)
                dorsal_surface_retouch_area = cv2.contourArea(dorsal_surface_retouch_outline) / (ref_Obj * ref_Obj)

                # give retouch positional argument
                if dorsal_surface_retouch_outline[1][1] <= int(topmost[1] + 5) and dorsal_surface_retouch_outline[-1][1] >= int(botmost[1] - 5):
                    sur_ret_pos_dor = 'complete'
                    relative_dorsal_surface_retouch_area = 1
                elif dorsal_surface_retouch_outline[1][1] <= int(topmost[1] + 5) and dorsal_surface_retouch_outline[-1][1] >= mid_lat[1]:
                    sur_ret_pos_dor = 'tip_and_medial'
                elif dorsal_surface_retouch_outline[1][1] <= int(topmost[1] + 5):
                    sur_ret_pos_dor = 'tip'
                elif dorsal_surface_retouch_outline[-1][1] >= int(botmost[1] - 5) and dorsal_surface_retouch_outline[1][1] <= mid_lat[1]:
                    sur_ret_pos_dor = 'base_and_medial'
                elif dorsal_surface_retouch_outline[-1][1] >= int(botmost[1] - 5):
                    sur_ret_pos_dor = 'base'
                elif dorsal_surface_retouch_outline[1][1] > int(topmost[1] + 5) and dorsal_surface_retouch_outline[-1][1] < int(botmost[1] - 5):
                    sur_ret_pos_dor = 'medial'

                # create retouch ID with site-name, artefact number, retouch-style, side of retouch, position of
                # retouch and absolute and relative length of retouch
                retouch_id.append(name + '_' + "%02d" % i + ',' + style + ',' + 'dorsal' + ',' + str(sur_ret_pos_dor) + ',' + str(1))

                # style output image
                cv2.putText(img, str('dorsal ') + sur_ret_pos_dor, (tl[0] - 80, dorsal_surface_retouch_outline[1][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                cv2.putText(img, str(style), (tl[0] - 80, dorsal_surface_retouch_outline[1][1] - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                cv2.putText(img, "{:.1f}".format(np.round(dorsal_surface_retouch_area, 2)), (tl[0] - 80, dorsal_surface_retouch_outline[1][1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

                # take the "ventral" list if they aren´t empty and calculate absolute and relative area
                if len(ventral) != 0:
                    ventral_surface_retouch = np.array(dorsal, dtype=np.int32)
                    ven_sur_ret = cv2.contourArea(ventral_surface_retouch) / (ref_Obj * ref_Obj)

                    # give retouch positional argument
                    if ventral_surface_retouch[1][1] <= int(topmost[1] + 5) and ventral_surface_retouch[-1][1] >= int(
                            botmost[1] - 5):
                        sur_ret_pos_ven = 'complete'
                    elif ventral_surface_retouch[1][1] <= int(topmost[1] + 5) and ventral_surface_retouch[-1][1] >= \
                            mid_lat[1]:
                        sur_ret_pos_ven = 'tip_and_medial'
                    elif ventral_surface_retouch[1][1] <= int(topmost[1] + 5):
                        sur_ret_pos_ven = 'tip'
                    elif ventral_surface_retouch[-1][1] >= int(botmost[1] - 5) and ventral_surface_retouch[1][1] <= \
                            mid_lat[1]:
                        sur_ret_pos_ven = 'base_and_medial'
                    elif ventral_surface_retouch[-1][1] >= int(botmost[1] - 5):
                        sur_ret_pos_ven = 'base'
                    elif ventral_surface_retouch[1][1] > int(topmost[1] + 5) and ventral_surface_retouch[-1][1] < int(
                            botmost[1] - 5):
                        sur_ret_pos_ven = 'medial'

                    # create retouch ID with site-name, artefact number, retouch-style, side of retouch, position of
                    # retouch and absolute and relative length of retouch
                    retouch_id.append(name + '_' + "%02d" % i + ',' + style + ',' + 'dorsal' + ',' + str(sur_ret_pos_ven) + ',' + str(np.round(ven_sur_ret, 2)))

                    # style output image
                    cv2.putText(img, str('ventral ') + sur_ret_pos_ven, (tr[0] + 10, ventral_surface_retouch[1][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                    cv2.putText(img, str(style), (tr[0] + 10, ventral_surface_retouch[1][1] - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
                    cv2.putText(img, "{:.1f}".format(np.round(ven_sur_ret, 2)), (tr[0] + 10, ventral_surface_retouch[1][1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

        i += 1  # enlarge ID for next artefact

    return retouch_id

def colormask_points(masked_img):

    # find contours of colored image using simple chain approx method
    cnts = cv2.findContours(masked_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    (cnts, _) = contours.sort_contours(cnts)

    # set list for colored points in all colorbars
    color_points_total = []

    for cnt in cnts:

        # set list for colored points in each colorbar
        color_points = []

        # approx colorbar
        contour_length = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * contour_length, True)

        # create bounding box
        x, y, w, h = cv2.boundingRect(approx)
        tl = x, y
        tr = x + w, y
        (tltrX, tltrY) = midpoint(tl, tr)
        mid = (tltrX, tltrY)

        # iterate through detected colorbars in y-direction
        ycount = 0
        for ptsY in range(h):
            ycount += 1
            yp = int(mid[0]), int(mid[1] + ycount)

            # append colored points for each colorbar
            color_points.append(yp)

        # append colored points of colorbars to a total list
        color_points_total.append(color_points)

    return color_points_total



### secondary functions ###

# extract name from directory
def set_imagename(filename):
    
    pathname, extension = os.path.splitext(filename)
    imagefilename = pathname.split('/')
    imagename = imagefilename[-1]
    
    return imagename

# create folder to store in data
def create_folder(name, directory):

    folder_directory = directory + name + "/"
    
    # check if folder already exists
    if os.path.exists(folder_directory):
        # if so, give warning signal
        duration = 300
        freq = 200
        winsound.Beep(freq, duration)

        # ask if user want´s to overwrite folder
        overwrite_command = input(name + '-' + 'Folder exists already! Do you want to overwrite [y] or expand [n] it? ')
        if overwrite_command.lower() == 'yes' or overwrite_command.lower() == 'y':
            print('Okay! File will be overwritten.')
            shutil.rmtree(directory + name, ignore_errors = True)
            os.mkdir(directory + name)
        if overwrite_command.lower() == 'no' or overwrite_command.lower() == 'n':
            print('Okay! File will be expanded.')
            
    # create folder
    else:
        os.mkdir(directory + name)

    return folder_directory

# export data to csv
def write_as_csv(data, filename):

    df = pd.DataFrame(data = data)
    df.to_csv(filename + '.csv', index=False, header=False)

# export retouch data to csv
def export_retouch_data(data, name):

    df = pd.DataFrame(data=data)
    # set header
    header = ['ArtefactID,style,side,position,length,relative_length']
    df.to_csv(name, index=False, header=header, quotechar=" ")      # ignore quotationmarks

# sound when program finished
def finish_sound():

    winsound.Beep(frequenz, duration)
    duration = 100
    frequenz = 300

# resize image for faster procession
def resize_image(img):

    #if int(img.shape[1]) != 2400:
    #    # resize to 2400 pixel width
    #    resize_factor = 2400 / int(img.shape[1])
        
    if int(img.shape[1]) > 2400 or int(img.shape[1]) < 1000:
        # resize to 2400 pixel width
        resize_factor = 2400 / int(img.shape[1])

    else:
        resize_factor = 1

    img_width = int(img.shape[1] * resize_factor)
    img_height = int(img.shape[0] * resize_factor)
    dsize = (img_width, img_height)
    resized_image = cv2.resize(img, dsize)

    return resized_image

# import and prepare image
def import_img(img_name):

    image = cv2.imread(img_name)
    img = resize_image(image)   # call resize function

    return load_img(img)

# load and prepare image
def load_img(img):
    
    # greyscale image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # white image to draw on later
    blank_img = 255 * np.ones_like(img)
    # threshold image using binary and otsu method
    thresh, img_thresh = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)

    # fill inner area of thresholded objects in order to improve contour detection
    img_floodfill = img_thresh.copy()
    h, w = img_thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img_floodfill, mask, (0, 0), 255)
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    img_out = img_thresh | img_floodfill_inv

    # find outmost contours without a chaine approx
    cnts = cv2.findContours(img_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0]
    # sort contours by x-value
    (cnts, _) = contours.sort_contours(cnts)

    return cnts, img, blank_img

# load and prepare color image
def import_color_img(color_img_name):

    color_image = cv2.imread(color_img_name)
    color_img = resize_image(color_image)     # call resize function

    # create images for each color using a range from lower to upper limit
    red_lower = np.array([0, 0, 240], dtype="uint8")
    red_upper = np.array([50, 50, 255], dtype="uint8")
    red_mask = cv2.inRange(color_img, red_lower, red_upper)

    blue_lower = np.array([150, 0, 0], dtype="uint8")
    blue_upper = np.array([255, 50, 50], dtype="uint8")
    blue_mask = cv2.inRange(color_img, blue_lower, blue_upper)

    green_lower = np.array([0, 150, 0], dtype="uint8")
    green_upper = np.array([50, 255, 50], dtype="uint8")
    green_mask = cv2.inRange(color_img, green_lower, green_upper)

    yellow_lower = np.array([0, 240, 240], dtype="uint8")
    yellow_upper = np.array([50, 255, 255], dtype="uint8")
    yellow_mask = cv2.inRange(color_img, yellow_lower, yellow_upper)

    purpel_lower = np.array([240, 0, 240], dtype="uint8")
    purpel_upper = np.array([255, 50, 255], dtype="uint8")
    purpel_mask = cv2.inRange(color_img, purpel_lower, purpel_upper)

    return red_mask, blue_mask, green_mask, yellow_mask, purpel_mask


### preperation and exportation of images for further processing

# image with rotated outlines
def rotated_outlines_img(cnts, img, img_name):

    area = 150
    i = 0 # set artefact ID
    
    # iterate threw artefact contours
    for cnt in cnts:

        # ignore objects with too small area
        if cv2.contourArea(cnt) < area:
            continue

        contour_length = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * contour_length, True)

        # draw reference object
        if i == 0:
            cv2.polylines(img, [approx], True, (0, 0, 0), 2)

        else:
            # create rotated minimum area rectangles
            min_rect = cv2.minAreaRect(cnt)
            rot_min_rect = cv2.boxPoints(min_rect)
            rot_min_rect = np.int0(rot_min_rect)
            rot_min_rect = perspective.order_points(rot_min_rect)
            (tl, tr, br, bl) = rot_min_rect

            # calculate angle of rotation of minimum area rectangle
            (tlblX, tlblY) = midpoint(tl, bl)
            midlef = int(tlblX), int(tlblY)
            (trbrX, trbrY) = midpoint(tr, br)
            midrig = int(trbrX), int(trbrY)

            angle = (angle_between(midlef, midrig))
            rotation_1 = 0
            # convert angle to use it for artefact rotation
            if 170 < angle < 180:
                rotation_1 = 180 - angle
            elif -170 > angle > -180:
                rotation_1 = -180 + abs(angle)
            elif 100 < angle < 120:
                rotation_1 = 90 - angle
            elif -100 > angle > -120:
                rotation_1 = -90 + (abs(angle))
            elif angle == 180:
                rotation_1 = 0

            # rotate artefact contours
            cnt_rot_1 = rotate_contour(cnt, rotation_1)

            # find extreme points in rotated artefact contours
            # and measure distance between them
            topmost_1 = tuple(cnt_rot_1[cnt_rot_1[:, :, 1].argmin()][0])
            lowmost_1 = tuple(cnt_rot_1[cnt_rot_1[:, :, 1].argmax()][0])
            length_1 = lowmost_1[1] - topmost_1[1]

            # align extreme points
            other_angle = abs(angle_between(lowmost_1, topmost_1)) - 90

            # sort out artefacts with are over 80% of minimum area rectangle
            # for they are considered to be of an almost rectangular shape
            w = dist.euclidean(tl, tr)
            h = dist.euclidean(tl, bl)
            art_area = cv2.contourArea(cnt)
            rect_area = w * h
            pc_area = np.round((int(art_area) / rect_area), 2) * 100

            if pc_area > 80:
                rotation_2 = 0

            # try aligned extreme point rotation
            elif pc_area < 80 and 0 < other_angle:
                rotation_2 = [-(other_angle)]
            elif pc_area < 80 and 0 > other_angle:
                rotation_2 = [abs(other_angle)]

            # rotate artefact contour
            cnt_rot_2 = rotate_contour(cnt_rot_1, rotation_2)

            # find extreme points in rotated artefact contours
            # and measure distance between them
            topmost_2 = tuple(cnt_rot_2[cnt_rot_2[:, :, 1].argmin()][0])
            lowmost_2 = tuple(cnt_rot_2[cnt_rot_2[:, :, 1].argmax()][0])
            length_2 = lowmost_2[1] - topmost_2[1]

            # test and draw contour with longer length to blank image
            if length_1 > length_2:
                cv2.drawContours(img, [cnt_rot_1], -1, (0, 0, 0), 2)
            else:
                cv2.drawContours(img, [cnt_rot_2], -1, (0, 0, 0), 2)
        
        # increase ID for next artefact
        i += 1
        
        # write image to folder directory
        cv2.imwrite(img_name, img)

    return img

# export image for retouch analysis preparation
def retouch_img_prep(img, cnts, processed_img_name):

    # iterate through artefact contours
    for cnt in cnts:
        contour_length = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * contour_length, True)

        # sort out reference object by its shape
        obj_cor = len(approx)
        if obj_cor == 4:
            continue

        # create bounding boxes for each artefact
        x, y, w, h = cv2.boundingRect(approx)
        btl = (x, y)
        bbl = (x, y + h)
        bbr = (x + w, y + h)
        btr = (x + w, y)

        # draw boxes beside the artefacts, where color can be filled in later
        cv2.rectangle(img, (int(btl[0] - 15), int(btl[1])), (int(bbl[0]) - 10, int(bbl[1])),
                      (100, 100, 100), 1)
        cv2.rectangle(img, (int(btr[0] + 15), int(btr[1])), (int(bbr[0]) + 10, int(bbr[1])),
                      (100, 100, 100), 1)
        cv2.rectangle(img, (int(btl[0] - 25), int(btl[1])), (int(bbl[0]) - 20, int(bbl[1])),
                      (100, 100, 100), 1)
        cv2.rectangle(img, (int(btr[0] + 25), int(btr[1])), (int(bbr[0]) + 20, int(bbr[1])),
                      (100, 100, 100), 1)

        # create legend for colorcodes
        cv2.putText(img, '= dorsal', (2270, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.circle(img, (2250, 20), 8, (0, 0, 255), -1)
        cv2.putText(img, '= bifacial', (2270, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.circle(img, (2250, 40), 8, (0, 255, 255), -1)
        cv2.putText(img, '= backing', (2270, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.circle(img, (2250, 60), 8, (255, 0, 0), -1)
        cv2.putText(img, '= ventral', (2270, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.circle(img, (2250, 80), 8, (0, 255, 0), -1)
        cv2.putText(img, '= surface', (2270, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.circle(img, (2250, 100), 8, (255, 0, 255), -1)

        # write image to folder directory
        cv2.imwrite(processed_img_name, img)

# export single rotated outlines for GMM
def export_single_rotated_outlines(img, cnts, name):

    ROI_number = 0  # set number for export image

    # iterate through artefact contours
    for cnt in cnts:
        contour_length = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * contour_length, True)

        # sort out reference object by its shape
        obj_cor = len(approx)
        if obj_cor == 4:
            continue

        # draw contours
        cv2.drawContours(img, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)

        # create bounding box
        x, y, w, h = cv2.boundingRect(approx)

        # define rectangle of interest based on bounding box
        ROI = img[(y - 10):y + (h + 10), (x - 10):x + (w + 10)]

        # increase number for next artefact
        ROI_number += 1

        # write single artefact images to folder directory
        cv2.imwrite(name.format("%02d" % ROI_number), ROI)


### helperfunctions

# returns midpoint between two points
def midpoint(ptA, ptB):

    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# returns point on contour right to the input point
def cnt_point_left(p1, c, w, help = 'p1: point for which the opposite point needs to be found'):

    xcount = 0

    # set testrange of 1.2 times the width for to check for points on contour
    testrange = int(w + w * 0.2)

    # iterating through all possible points in x-direction
    for pts in range(testrange):
        xcount += 1
        pts = int((p1[0] - 15) + xcount), int(p1[1])

        # return the first one on contour
        ptoncnt = cv2.pointPolygonTest(c, pts, False)
        if ptoncnt == 0:
            return pts

# returns point on contour left to the input point
def cnt_point_right(p1, c, w, help = 'p1: point for which the opposite point needs to be found'):

    xcount = 0

    # set testrange of 1.2 times the width for to check for points on contour
    testrange = int(w + w * 0.2)

    # iterating through all possible points in negativ x-direction
    for pts in range(testrange):
        xcount -= 1
        pts = int((p1[0] + 15) + xcount), int(p1[1])

        # return the first one on contour
        ptoncnt = cv2.pointPolygonTest(c, pts, False)
        if ptoncnt == 0:
            return pts

# returns point on contour below the input point
def cnt_point_under(p1, cnt, h):

    ycount = 0

    # set testrange of 1.2 times the height for to check for points on contour
    testrange = int(h + h * 0.2)

    # iterating through all possible points in y-direction
    for pts in range(testrange):
        ycount += 1
        pts = int(p1[0]), int((p1[1] - 10) + ycount)

        # return the first one on contour
        ptoncnt = cv2.pointPolygonTest(cnt, pts, False)
        if ptoncnt == 0:
            return pts

# returns point on contour above the input point
def cnt_point_above(p1, cnt, h):

    ycount = 0

    # set testrange of 1.2 times the height for to check for points on contour
    testrange = int(h + h * 0.2)

    # iterating through all possible points in negativ y-direction
    for pts in range(testrange):
        ycount -= 1
        pts = int(p1[0]), int((p1[1] + 10) + ycount)

        # return the first one on contour
        ptoncnt = cv2.pointPolygonTest(cnt, pts, False)
        if ptoncnt == 0:
            return pts

# returns points and length of the maximum width
def widthpoints(p1, p2, h, cnt, w, ref_Obj):

    ycount = 0

    # set testrange for to check for widest part
    testrangeY = int(h)

    # create list for later return
    max_width = np.array([[1, 2, 3]], dtype=object)

    # iterate through possible points in y-direction left and right of artefact contour
    for ptsY in range(testrangeY):
        ycount += 1

        # find point on contour for each of them
        yleft = int(p1[0] - 5), int(p1[1] + ycount)
        ptsleft = cnt_point_left(yleft, cnt, w)
        if ptsleft == None:
            continue
        yright = int(p2[0] + 5), int(p2[1] + ycount)
        ptsright = cnt_point_right(yright, cnt, w)
        if ptsright == None:
            continue

        # calculate width between oblique points on contour
        wid = dist.euclidean(ptsleft, ptsright) / ref_Obj
        # merge width an points together
        array_temp = np.array([[wid, ptsleft, ptsright]], dtype = object)
        max_width = np.concatenate((max_width, array_temp), dtype = object)
        # filter widest width
        max_filter = max_width[max_width[:, 0] == max(max_width[:, 0])]
        # in case more then one pair of points delivers widest width the medien pair is chosen
        max_median = int(np.round(len(max_filter) / 2, 0))

    return max_filter[max_median, 1], max_filter[max_median, 2], max_filter[max_median, 0]

# returns points and length of maximum length
def lengthpoints(p1,p2, h, cnt, w, ref_Obj):

    xcount = 0

    # set testrange for to check for widest part
    testrangeX = int(w)

    # create list for later return
    max_length = np.array([[1, 2, 3]], dtype=object)

    # iterate through possible points in x-direction above and below artefact contour
    for ptsY in range(testrangeX):
        xcount += 1

        # find point on contour for each of them
        xtop = int(p1[0] + xcount), int(p1[1] - 5)
        ptstop = cnt_point_under(xtop, cnt, h)
        if ptstop == None:
            continue

        xbot = int(p2[0] + xcount), int(p2[1] + 5)
        ptsbot = cnt_point_above(xbot, cnt, h)
        if ptsbot == None:
            continue

        # calculate length between oblique points on contour
        leng = dist.euclidean(ptstop, ptsbot) / ref_Obj
        # merge width an points together
        array_temp = np.array([[leng, ptstop, ptsbot]], dtype = object)
        max_length = np.concatenate((max_length, array_temp), dtype = object)
        # filter longest length
        max_filter = max_length[max_length[:, 0] == max(max_length[:, 0])]
        # in case more then one pair of points delivers widest width the medien pair is chosen
        max_median = int(np.round(len(max_filter) / 2, 0))

    return max_filter[max_median, 1], max_filter[max_median, 2], max_filter[max_median, 0]

# return position of the maximum width defined by five zones
def max_length_pos(p1, w, p2, help = 'p1: toppoint, p2: topleft'):

    # define five zones by width started by the leftmost
    zone_1 = int(p2[0] + (w * 0.2)), int(p2[1])
    zone_2 = int(p2[0] + (w * 0.4)), int(p2[1])
    zone_3 = int(p2[0] + (w * 0.6)), int(p2[1])
    zone_4 = int(p2[0] + (w * 0.8)), int(p2[1])
    zone_5 = int(p2[0] + w), int(p2[1])

    # check whether point is in between zone points
    if p2[0] <= p1[0] < zone_1[0]:
        max_len_pos = 1
    if zone_1[0] <= p1[0] < zone_2[0]:
        max_len_pos = 2
    if zone_2[0] <= p1[0] < zone_3[0]:
        max_len_pos = 3
    if zone_3[0] <= p1[0] < zone_4[0]:
        max_len_pos = 4
    if zone_4[0] <= p1[0] < zone_5[0]:
        max_len_pos = 5

    return max_len_pos

# return position of the maximum length defined by five zones
def max_width_pos(p1, h, p2):

    # define five zones by height started from the lowest
    zone_1 = int(p2[0]), int(p2[1] - (h * 0.2))
    zone_2 = int(p2[0]), int(p2[1] - (h * 0.4))
    zone_3 = int(p2[0]), int(p2[1] - (h * 0.6))
    zone_4 = int(p2[0]), int(p2[1] - (h * 0.8))
    zone_5 = int(p2[0]), int(p2[1] - h)

    # check if point ist in between zone points
    if p2[1] >= p1[1] > zone_1[1]:
        max_wid_pos = 1
    if zone_1[1] >= p1[1] > zone_2[1]:
        max_wid_pos = 2
    if zone_2[1] >= p1[1] > zone_3[1]:
        max_wid_pos = 3
    if zone_3[1] >= p1[1] > zone_4[1]:
        max_wid_pos = 4
    if zone_4[1] >= p1[1] > zone_5[1]:
        max_wid_pos = 5

    return max_wid_pos

# returns a set of 16 equally distributed points started with the bottom left going clockwise
def checkpoints(img, pt1, pt2, pt3, pt4, pt5, pt6, topmost, botmost, cnt, w, h, pc_area, ref_Obj, help = 'left extreme, tl, bl, right extreme, tr, br'):

    # use percentage area to distinguish between pointy artefacts
    # and artefact of a more rectangular shape
    # if percentage area is smaller then 85, artefact is considered
    # to be more or less of a pointy shape
    if pc_area < 85:
        # find regularly placed points using midpoint helperfunction
        # and find associated point with the same y-value on contour
        # using cnt_point functions
        (pt1pt2X, pt1pt2Y) = midpoint(pt1, pt2)
        uml = int(pt1pt2X), int(pt1pt2Y)
        upper_mid_lef = cnt_point_left(uml, cnt, w)

        (umpt2X, umpt2Y) = midpoint(uml, (pt2[0], pt2[1]))
        uul = int(umpt2X), int(umpt2Y)
        upper_upp_lef = cnt_point_left(uul, cnt, w)

        (umpt1X, umpt1Y) = midpoint(uml, (pt1[0], pt1[1]))
        ull = int(umpt1X), int(umpt1Y)
        upper_low_lef = cnt_point_left(ull, cnt, w)

        (pt4pt5X, pt4pt5Y) = midpoint(pt4, pt5)
        umr = int(pt4pt5X), int(pt4pt5Y)
        upper_mid_rig = cnt_point_right(umr, cnt, w)

        (umpt5X, umpt5Y) = midpoint(umr, (pt5[0], pt5[1]))
        uur = int(umpt5X), int(umpt5Y)
        upper_upp_rig = cnt_point_right(uur, cnt, w)

        (umpt4X, umpt4Y) = midpoint(umr, (pt4[0], pt4[1]))
        ulr = int(umpt4X), int(umpt4Y)
        upper_low_rig = cnt_point_right(ulr, cnt, w)

        (pt1pt3X, pt1pt3Y) = midpoint(pt1, pt3)
        lml = int(pt1pt3X), int(pt1pt3Y)
        lower_mid_lef = cnt_point_left(lml, cnt, w)

        (lmpt3X, lmpt3Y) = midpoint(lml, pt3)
        lll = int(lmpt3X), int(lmpt3Y)
        lower_low_lef = cnt_point_left(lll, cnt, w)

        (lmpt1X, lmpt1Y) = midpoint(lml, (pt1[0], pt1[1]))
        lul = int(lmpt1X), int(lmpt1Y)
        lower_upp_lef = cnt_point_left(lul, cnt, w)

        (pt4pt6X, pt4pt6Y) = midpoint(pt4, pt6)
        lmr = int(pt4pt6X), int(pt4pt6Y)
        lower_mid_rig = cnt_point_right(lmr, cnt, w)

        (lmpt6X, lmpt6Y) = midpoint(lmr, (pt6[0], pt6[1]))
        llr = int(lmpt6X), int(lmpt6Y)
        lower_low_rig = cnt_point_right((int(llr[0]), int(llr[1])), cnt, w)

        (lmpt4X, lmpt4Y) = midpoint(lmr, (pt4[0], pt4[1]))
        lur = int(lmpt4X), int(lmpt4Y)
        lower_upp_rig = cnt_point_right(lur, cnt, w)

        # create list containing all points
        cpoints = [lower_low_lef, lower_mid_lef, lower_upp_lef,
                   pt1,
                   upper_low_lef, upper_mid_lef, upper_upp_lef,
                   topmost,
                   upper_upp_rig, upper_mid_rig, upper_low_rig,
                   pt4,
                   lower_upp_rig, lower_mid_rig, lower_low_rig,
                   botmost]

    # if shape is considered to be more of a rectangular shape
    # three points are set at top and bottom and five points are
    # set left and right of artefact including widest part points
    else:
        # horizontal points
        zl1 = int(pt3[0]), int(pt3[1] - (h * 0.2))
        zl2 = int(pt3[0]), int(pt3[1] - (h * 0.4))
        zl3 = int(pt3[0]), int(pt3[1] - (h * 0.6))
        zl4 = int(pt3[0]), int(pt3[1] - (h * 0.8))

        zone_1_l_point = midpoint(pt3, zl1)
        zone_2_l_point = midpoint(zl1, zl2)
        zone_3_l_point = midpoint(zl2, zl3)
        zone_4_l_point = midpoint(zl3, zl4)
        zone_5_l_point = midpoint(zl4, pt2)

        zone_1_l_point_cnt = cnt_point_left(zone_1_l_point, cnt, w)
        zone_2_l_point_cnt = cnt_point_left(zone_2_l_point, cnt, w)
        zone_3_l_point_cnt = cnt_point_left(zone_3_l_point, cnt, w)
        zone_4_l_point_cnt = cnt_point_left(zone_4_l_point, cnt, w)
        zone_5_l_point_cnt = cnt_point_left(zone_5_l_point, cnt, w)

        zr1 = int(pt6[0]), int(pt6[1] - (h * 0.2))
        zr2 = int(pt6[0]), int(pt6[1] - (h * 0.4))
        zr3 = int(pt6[0]), int(pt6[1] - (h * 0.6))
        zr4 = int(pt6[0]), int(pt6[1] - (h * 0.8))

        zone_1_r_point = midpoint(pt6, zr1)
        zone_2_r_point = midpoint(zr1, zr2)
        zone_3_r_point = midpoint(zr2, zr3)
        zone_4_r_point = midpoint(zr3, zr4)
        zone_5_r_point = midpoint(zr4, pt5)

        zone_1_r_point_cnt = cnt_point_right(zone_1_r_point, cnt, w)
        zone_2_r_point_cnt = cnt_point_right(zone_2_r_point, cnt, w)
        zone_3_r_point_cnt = cnt_point_right(zone_3_r_point, cnt, w)
        zone_4_r_point_cnt = cnt_point_right(zone_4_r_point, cnt, w)
        zone_5_r_point_cnt = cnt_point_right(zone_5_r_point, cnt, w)

        # vertical points
        zt1 = int(pt2[0] + (w * 0.33)), int(pt2[1])
        zt2 = int(pt2[0] + (w * 0.66)), int(pt2[1])
        zt3 = int(pt5[0]), int(pt5[1])

        zone_1_t_point = midpoint(pt2, zt1)
        zone_2_t_point = midpoint(zt1, zt2)
        zone_3_t_point = midpoint(zt2, zt3)

        zone_1_t_point_cnt = cnt_point_under(zone_1_t_point, cnt, h)
        zone_2_t_point_cnt = cnt_point_under(zone_2_t_point, cnt, h)
        zone_3_t_point_cnt = cnt_point_under(zone_3_t_point, cnt, h)

        zb1 = int(pt3[0] + (w * 0.3333)), int(pt3[1])
        zb2 = int(pt3[0] + (w * 0.6666)), int(pt3[1])
        zb3 = int(pt6[0]), int(pt6[1])

        zone_1_b_point = midpoint(pt3, zb1)
        zone_2_b_point = midpoint(zb1, zb2)
        zone_3_b_point = midpoint(zb2, zb3)

        zone_1_b_point_cnt = cnt_point_above(zone_1_b_point, cnt, h)
        zone_2_b_point_cnt = cnt_point_above(zone_2_b_point, cnt, h)
        zone_3_b_point_cnt = cnt_point_above(zone_3_b_point, cnt, h)

        if pt1[1] > zl1[1]:
            cpoints = [pt1, zone_2_l_point_cnt, zone_3_l_point_cnt, zone_4_l_point_cnt, zone_5_l_point_cnt,
                       zone_1_t_point_cnt, zone_2_t_point_cnt, zone_3_t_point_cnt,
                       zone_5_r_point_cnt, zone_4_r_point_cnt, zone_3_r_point_cnt, zone_2_r_point_cnt, pt4,
                       zone_3_b_point_cnt, zone_2_b_point_cnt, zone_1_b_point_cnt]
        if zl1[1] > pt1[1] > zl2[1]:
            cpoints = [zone_1_l_point_cnt, pt1, zone_3_l_point_cnt, zone_4_l_point_cnt, zone_5_l_point_cnt,
                       zone_1_t_point_cnt, zone_2_t_point_cnt, zone_3_t_point_cnt,
                       zone_5_r_point_cnt, zone_4_r_point_cnt, zone_3_r_point_cnt, pt4, zone_1_r_point_cnt,
                       zone_3_b_point_cnt, zone_2_b_point_cnt, zone_1_b_point_cnt]
        if zl2[1] > pt1[1] > zl3[1]:
            cpoints = [zone_1_l_point_cnt, zone_2_l_point_cnt, pt1, zone_4_l_point_cnt, zone_5_l_point_cnt,
                       zone_1_t_point_cnt, zone_2_t_point_cnt, zone_3_t_point_cnt,
                       zone_5_r_point_cnt, zone_4_r_point_cnt, pt4, zone_2_r_point_cnt, zone_1_r_point_cnt,
                       zone_3_b_point_cnt, zone_2_b_point_cnt, zone_1_b_point_cnt]
        if zl3[1] > pt1[1] > zl4[1]:
            cpoints = [zone_1_l_point_cnt, zone_2_l_point_cnt, zone_3_l_point_cnt, pt1, zone_5_l_point_cnt,
                       zone_1_t_point_cnt, zone_2_t_point_cnt, zone_3_t_point_cnt,
                       zone_5_r_point_cnt, pt4, zone_3_r_point_cnt, zone_2_r_point_cnt, zone_1_r_point_cnt,
                       zone_3_b_point_cnt, zone_2_b_point_cnt, zone_1_b_point_cnt]
        if zl4[1] > pt1[1]:
            cpoints = [zone_1_l_point_cnt, zone_2_l_point_cnt, zone_3_l_point_cnt, zone_4_l_point_cnt, pt1,
                       zone_1_t_point_cnt, zone_2_t_point_cnt, zone_3_t_point_cnt,
                       pt4, zone_4_r_point_cnt, zone_3_r_point_cnt, zone_2_r_point_cnt, zone_1_r_point_cnt,
                       zone_3_b_point_cnt, zone_2_b_point_cnt, zone_1_b_point_cnt]

    # draw points on image
    for poi in cpoints:
        cv2.circle(img, poi, 5, (50, 150, 200), -1)

    return cpoints

# returns tree widths for each upper and lower part starting with the lowest
def part_widths(img, cpoints, ref_Obj, pc_area, lef):

    if pc_area < 85:
        lower_low_lef, lower_mid_lef, lower_upp_lef, left, upper_low_lef, upper_mid_lef, upper_upp_lef, top, upper_upp_rig, upper_mid_rig, upper_low_rig, right, lower_upp_rig, lower_mid_rig, lower_low_rig, bot = cpoints

        lower_low_width = int(dist.euclidean(lower_low_lef, lower_low_rig) / ref_Obj)
        lower_mid_width = int(dist.euclidean(lower_mid_lef, lower_mid_rig) / ref_Obj)
        lower_up_width = int(dist.euclidean(lower_upp_lef, lower_upp_rig) / ref_Obj)
        upper_low_width = int(dist.euclidean(upper_low_lef, upper_low_rig) / ref_Obj)
        upper_mid_width = int(dist.euclidean(upper_mid_lef, upper_mid_rig) / ref_Obj)
        upper_up_width = int(dist.euclidean(upper_upp_lef, upper_upp_rig) / ref_Obj)

        cv2.putText(img, "{:.0f}".format(lower_low_width), (left[0] - 30, lower_low_lef[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (100, 0, 0), 2)
        cv2.putText(img, "{:.0f}".format(lower_mid_width), (left[0] - 30, lower_mid_lef[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (100, 0, 0), 2)
        cv2.putText(img, "{:.0f}".format(lower_up_width), (left[0] - 30, lower_upp_lef[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (100, 0, 0), 2)
        cv2.putText(img, "{:.0f}".format(upper_low_width), (left[0] - 30, upper_low_lef[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (100, 0, 0), 2)
        cv2.putText(img, "{:.0f}".format(upper_mid_width), (left[0] - 30, upper_mid_lef[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (100, 0, 0), 2)
        cv2.putText(img, "{:.0f}".format(upper_up_width), (left[0] - 30, upper_upp_lef[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (100, 0, 0), 2)

        cv2.line(img, lower_low_lef, lower_low_rig, (100, 0, 0), 2)
        cv2.line(img, lower_mid_lef, lower_mid_rig, (100, 0, 0), 2)
        cv2.line(img, lower_upp_lef, lower_upp_rig, (100, 0, 0), 2)
        cv2.line(img, upper_low_lef, upper_low_rig, (100, 0, 0), 2)
        cv2.line(img, upper_mid_lef, upper_mid_rig, (100, 0, 0), 2)
        cv2.line(img, upper_upp_lef, upper_upp_rig, (100, 0, 0), 2)

        widest_width = np.round(dist.euclidean(left, right) / ref_Obj, 1)

        part_widths = [lower_low_width, lower_mid_width, lower_up_width, widest_width, upper_low_width, upper_mid_width, upper_up_width]

    else:
        lower_low_lef, lower_mid_lef, mid_lef, upper_mid_lef, upper_upp_lef, top_lef, top_mid, top_rig, upper_upp_rig, upper_mid_rig, mid_rig, lower_mid_rig, lower_low_rig, bot_rig, bot_mid, bot_lef = cpoints

        upper_up_width = int(dist.euclidean(upper_upp_lef, upper_upp_rig) / ref_Obj)
        upper_mid_width = int(dist.euclidean(upper_mid_lef, upper_mid_rig) / ref_Obj)
        mid_width = int(dist.euclidean(mid_lef, mid_rig) / ref_Obj)
        lower_mid_width = int(dist.euclidean(lower_mid_lef, lower_mid_rig) / ref_Obj)
        lower_low_width = int(dist.euclidean(lower_low_lef, lower_low_rig) / ref_Obj)

        if upper_upp_lef != lef:
            cv2.line(img, upper_upp_lef, upper_upp_rig, (100, 0, 0), 2)
            cv2.putText(img, "{:.0f}".format(upper_up_width), (mid_lef[0] - 30, upper_upp_lef[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (100, 0, 0), 2)
        if upper_mid_lef != lef:
            cv2.line(img, upper_mid_lef, upper_mid_rig, (100, 0, 0), 2)
            cv2.putText(img, "{:.0f}".format(upper_mid_width), (mid_lef[0] - 30, upper_mid_lef[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (100, 0, 0), 2)
        if mid_lef != lef:
            cv2.line(img, mid_lef, mid_rig, (100, 0, 0), 2)
            cv2.putText(img, "{:.0f}".format(mid_width), (mid_lef[0] - 30, mid_lef[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (100, 0, 0), 2)
        if lower_mid_lef != lef:
            cv2.line(img, lower_mid_lef, lower_mid_rig, (100, 0, 0), 2)
            cv2.putText(img, "{:.0f}".format(lower_mid_width), (mid_lef[0] - 30, lower_mid_lef[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (100, 0, 0), 2)
        if lower_low_lef != lef:
            cv2.line(img, lower_low_lef, lower_low_rig, (100, 0, 0), 2)
            cv2.putText(img, "{:.0f}".format(lower_low_width), (mid_lef[0] - 30, lower_low_lef[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (100, 0, 0), 2)

        part_widths = [lower_low_width, lower_low_width, lower_mid_width, mid_width, upper_mid_width, upper_up_width, 0]

    return part_widths

# returns the lengths for five zones starting from the left
def length_zone_measure(img, pt1, pt2, pt3, pt4, cnt, w, h, ref_Obj, pc_area, cpoints, help='top, tl, bl, bot'):

    if pc_area < 85:
        zt1 = int(pt2[0] + (w * 0.2)), int(pt2[1])
        zt2 = int(pt2[0] + (w * 0.4)), int(pt2[1])
        zt3 = int(pt2[0] + (w * 0.6)), int(pt2[1])
        zt4 = int(pt2[0] + (w * 0.8)), int(pt2[1])
        zt5 = int(pt2[0] + w), int(pt2[1])

        zone_1_t_point = midpoint(pt2, zt1)
        zone_2_t_point = midpoint(zt1, zt2)
        zone_3_t_point = midpoint(zt2, zt3)
        zone_4_t_point = midpoint(zt3, zt4)
        zone_5_t_point = midpoint(zt4, zt5)

        zone_1_t_point_cnt = cnt_point_under(zone_1_t_point, cnt, h)
        zone_2_t_point_cnt = cnt_point_under(zone_2_t_point, cnt, h)
        zone_3_t_point_cnt = cnt_point_under(zone_3_t_point, cnt, h)
        zone_4_t_point_cnt = cnt_point_under(zone_4_t_point, cnt, h)
        zone_5_t_point_cnt = cnt_point_under(zone_5_t_point, cnt, h)

        zb1 = int(pt3[0] + (w * 0.2)), int(pt3[1])
        zb2 = int(pt3[0] + (w * 0.4)), int(pt3[1])
        zb3 = int(pt3[0] + (w * 0.6)), int(pt3[1])
        zb4 = int(pt3[0] + (w * 0.8)), int(pt3[1])
        zb5 = int(pt3[0] + w), int(pt3[1])

        zone_1_b_point = midpoint(pt3, zb1)
        zone_2_b_point = midpoint(zb1, zb2)
        zone_3_b_point = midpoint(zb2, zb3)
        zone_4_b_point = midpoint(zb3, zb4)
        zone_5_b_point = midpoint(zb4, zb5)

        zone_1_b_point_cnt = cnt_point_above(zone_1_b_point, cnt, h)
        zone_2_b_point_cnt = cnt_point_above(zone_2_b_point, cnt, h)
        zone_3_b_point_cnt = cnt_point_above(zone_3_b_point, cnt, h)
        zone_4_b_point_cnt = cnt_point_above(zone_4_b_point, cnt, h)
        zone_5_b_point_cnt = cnt_point_above(zone_5_b_point, cnt, h)

        longest_length = np.round(dist.euclidean(pt1, pt4) / ref_Obj, 1)

        if pt2[0] <= pt1[0] < zt1[0]:
            cv2.line(img, zone_2_t_point_cnt, zone_2_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_3_t_point_cnt, zone_3_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_4_t_point_cnt, zone_4_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_5_t_point_cnt, zone_5_b_point_cnt, (100, 0, 0), 2)

            zone2_length = int(dist.euclidean(zone_2_t_point_cnt, zone_2_b_point_cnt) / ref_Obj)
            zone3_length = int(dist.euclidean(zone_3_t_point_cnt, zone_3_b_point_cnt) / ref_Obj)
            zone4_length = int(dist.euclidean(zone_4_t_point_cnt, zone_4_b_point_cnt) / ref_Obj)
            zone5_length = int(dist.euclidean(zone_5_t_point_cnt, zone_5_b_point_cnt) / ref_Obj)

            cv2.putText(img, "{:.0f}".format(zone2_length) + str(' ') + "{:.0f}".format(zone3_length) + str(' ') + "{:.0f}".format(zone4_length) +
                        str(' ') + "{:.0f}".format(zone5_length), (zone_1_b_point_cnt[0] - 25, pt4[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 0, 0), 2)

            zone_lengths = [longest_length, zone2_length, zone3_length, zone4_length, zone5_length]

        if zt1[0] <= pt1[0] < zt2[0]:
            cv2.line(img, zone_1_t_point_cnt, zone_1_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_3_t_point_cnt, zone_3_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_4_t_point_cnt, zone_4_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_5_t_point_cnt, zone_5_b_point_cnt, (100, 0, 0), 2)

            zone1_length = int(dist.euclidean(zone_1_t_point_cnt, zone_1_b_point_cnt) / ref_Obj)
            zone3_length = int(dist.euclidean(zone_3_t_point_cnt, zone_3_b_point_cnt) / ref_Obj)
            zone4_length = int(dist.euclidean(zone_4_t_point_cnt, zone_4_b_point_cnt) / ref_Obj)
            zone5_length = int(dist.euclidean(zone_5_t_point_cnt, zone_5_b_point_cnt) / ref_Obj)

            cv2.putText(img, "{:.0f}".format(zone1_length) + str(' ') + "{:.0f}".format(zone3_length) + str(' ') + "{:.0f}".format(zone4_length) +
                        str(' ') + "{:.0f}".format(zone5_length), (zone_1_b_point_cnt[0] - 25, pt4[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 0, 0), 2)

            zone_lengths = [zone1_length, longest_length, zone3_length, zone4_length, zone5_length]

        if zt2[0] <= pt1[0] < zt3[0]:
            cv2.line(img, zone_1_t_point_cnt, zone_1_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_2_t_point_cnt, zone_2_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_4_t_point_cnt, zone_4_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_5_t_point_cnt, zone_5_b_point_cnt, (100, 0, 0), 2)

            zone1_length = int(dist.euclidean(zone_1_t_point_cnt, zone_1_b_point_cnt) / ref_Obj)
            zone2_length = int(dist.euclidean(zone_2_t_point_cnt, zone_2_b_point_cnt) / ref_Obj)
            zone4_length = int(dist.euclidean(zone_4_t_point_cnt, zone_4_b_point_cnt) / ref_Obj)
            zone5_length = int(dist.euclidean(zone_5_t_point_cnt, zone_5_b_point_cnt) / ref_Obj)

            cv2.putText(img, "{:.0f}".format(zone1_length) + str(' ') + "{:.0f}".format(zone2_length) + str(' ') + "{:.0f}".format(zone4_length) +
                        str(' ') + "{:.0f}".format(zone5_length), (zone_1_b_point_cnt[0] - 25, pt4[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 0, 0), 2)

            zone_lengths = [zone1_length, zone2_length, longest_length, zone4_length, zone5_length]

        if zt3[0] <= pt1[0] < zt4[0]:
            cv2.line(img, zone_1_t_point_cnt, zone_1_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_2_t_point_cnt, zone_2_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_3_t_point_cnt, zone_3_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_5_t_point_cnt, zone_5_b_point_cnt, (100, 0, 0), 2)

            zone1_length = int(dist.euclidean(zone_1_t_point_cnt, zone_1_b_point_cnt) / ref_Obj)
            zone2_length = int(dist.euclidean(zone_2_t_point_cnt, zone_2_b_point_cnt) / ref_Obj)
            zone3_length = int(dist.euclidean(zone_3_t_point_cnt, zone_3_b_point_cnt) / ref_Obj)
            zone5_length = int(dist.euclidean(zone_5_t_point_cnt, zone_5_b_point_cnt) / ref_Obj)

            cv2.putText(img, "{:.0f}".format(zone1_length) + str(' ') + "{:.0f}".format(zone2_length) + str(' ') + "{:.0f}".format(zone3_length) +
                        str(' ') + "{:.0f}".format(zone5_length), (zone_1_b_point_cnt[0] - 25, pt4[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 0, 0), 2)

            zone_lengths = [zone1_length, zone2_length, zone3_length, longest_length, zone5_length]

        if zt4[0] <= pt1[0] < zt5[0]:
            cv2.line(img, zone_1_t_point_cnt, zone_1_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_2_t_point_cnt, zone_2_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_3_t_point_cnt, zone_3_b_point_cnt, (100, 0, 0), 2)
            cv2.line(img, zone_4_t_point_cnt, zone_4_b_point_cnt, (100, 0, 0), 2)

            zone1_length = int(dist.euclidean(zone_1_t_point_cnt, zone_1_b_point_cnt) / ref_Obj)
            zone2_length = int(dist.euclidean(zone_2_t_point_cnt, zone_2_b_point_cnt) / ref_Obj)
            zone3_length = int(dist.euclidean(zone_3_t_point_cnt, zone_3_b_point_cnt) / ref_Obj)
            zone4_length = int(dist.euclidean(zone_4_t_point_cnt, zone_4_b_point_cnt) / ref_Obj)

            cv2.putText(img, "{:.0f}".format(zone1_length) + str(' ') + "{:.0f}".format(zone2_length) + str(' ') + "{:.0f}".format(zone3_length) +
                        str(' ') + "{:.0f}".format(zone4_length), (zone_1_b_point_cnt[0] - 25, pt4[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 0, 0), 2)

            zone_lengths = [zone1_length, zone2_length, zone3_length, zone4_length, longest_length]

    else:
        lower_low_lef, lower_mid_lef, mid_lef, upper_mid_lef, upper_upp_lef, top_lef, top_mid, top_rig, upper_upp_rig, upper_mid_rig, mid_rig, lower_mid_rig, lower_low_rig, bot_rig, bot_mid, bot_lef = cpoints

        cv2.line(img, top_lef, bot_lef, (100, 0, 0), 2)
        cv2.line(img, top_mid, bot_mid, (100, 0, 0), 2)
        cv2.line(img, top_rig, bot_rig, (100, 0, 0), 2)

        zone1_length = int(dist.euclidean(top_lef, bot_lef) / ref_Obj)
        zone2_length = int(dist.euclidean(top_mid, bot_mid) / ref_Obj)
        zone3_length = int(dist.euclidean(top_rig, bot_rig) / ref_Obj)
        longest_length = int(dist.euclidean(pt1, pt4) / ref_Obj)

        cv2.putText(img, "{:.0f}".format(zone1_length) + str(' ') + "{:.0f}".format(zone2_length) + str(' ') + "{:.0f}".format(zone3_length),
                    (bot_lef[0] - 25, pt4[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 0, 0), 2)

        if pt1[0] < top_lef[0]:
            zone_lengths = [longest_length, zone1_length, zone2_length, zone3_length, 0]
        if top_lef[0] <= pt1[0] < top_mid[0]:
            zone_lengths = [zone1_length, longest_length, zone2_length, zone3_length, 0]
        if top_mid[0] <= pt1[0] < top_rig[0]:
            zone_lengths = [zone1_length, zone2_length, longest_length, zone3_length, 0]
        if pt1[0] > top_rig[0]:
            zone_lengths = [zone1_length, zone2_length, zone3_length, longest_length, 0]

    return zone_lengths

# measures the angle for every checkpoint
def angles(point_list):

    all_angles = []

    for index, elem in enumerate(point_list):
        
        if index + 1 < len(point_list):
            pre = (point_list[index - 1])
            cur = (elem)
            nex = (point_list[index + 1])
        else:
            pre = (point_list[index -1])
            cur = (elem)
            nex = (point_list[0])

        ba = pre[0] - cur[0], pre[1] - cur[1]
        bc = nex[0] - cur[0], nex[1] - cur[1]

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(cosine_angle))
        all_angles.append(angle)

    return all_angles

# helperfunction for angle
def angle_between(p1, p2):

    radians = math.atan2(p1[1]-p2[1], p1[0]-p2[0])
    degrees = math.degrees(radians)
    return degrees

# calculates and returns the vector between MP und CM
def midpoint_offset(img, cnt, tl, br, ref_Obj):

    # find center of mass (CM)
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    CM = (cX, cY)

    # find geometric midpoint (MP)
    (tlbrX, tlbrY) = midpoint(tl, br)
    MP = (int(tlbrX), int(tlbrY))

    # in order to create a vector, showing the offset of CM from MP, we:

    # calculate distance between MP and CM in x-direction
    # so that if CM is left to MP, the value turns negative
    x_offset = np.round((CM[0] - MP[0]) / ref_Obj, 1)

    # calculate distance between MP and CM in y-direction
    # so that, if CM is below the MP, the value turns negative.
    # Because we are working with a matrix here, y is less when it is higher.
    y_offset = np.round((MP[1] - CM[1]) / ref_Obj, 1)

    # calculate distance between MP and CM
    distance = np.round((dist.euclidean(MP, CM)) / ref_Obj, 1)

    # calculate ange between MP and CM
    angle = angle_between(MP, CM)

    # recalculate angles in circular way as "direction", started by 0 at left
    if angle >= 0:
        new_angle = np.round(angle, 1)
    elif angle < 0:
        new_angle = np.round(angle + 360, 1)

    # set 0.5mm buffer value to define "no offset"
    buffer = (ref_Obj * 0.05)

    if distance < buffer:
        direction_class = 'no offset'

    # reclassify angle to eight direction classes
    elif distance >= buffer and 77.5 < new_angle <= 122.5:
        direction_class = 'front'
    elif distance >= buffer and 122.5 < new_angle <= 167.5:
        direction_class = 'front right'
    elif distance >= buffer and 167.5 < new_angle <= 202.5:
        direction_class = 'right'
    elif distance >= buffer and 202.5 < new_angle <= 247.5:
        direction_class = 'back right'
    elif distance >= buffer and 247.5 < new_angle <= 292.5:
        direction_class = 'back'
    elif distance >= buffer and 292.5 < new_angle <= 337.5:
        direction_class = 'back left'
    elif distance >= buffer and new_angle > 337.5:
        direction_class = 'left'
    elif distance >= buffer and new_angle <= 22.5:       # 'left' needs to be assigned double, because the values exceed 0
        direction_class = 'left'
    elif distance >= buffer and 22.5 < new_angle <= 77.5:
        direction_class = 'front left'

    # style output image
    cv2.line(img, MP, CM, (0, 0, 0), 2)
    cv2.circle(img, MP, 4, (0, 0, 150), -1)
    cv2.circle(img, (cX, cY), 4, (0, 150, 0), -1)

    return x_offset, y_offset #, distance, direction_class, new_angle


### helperfunctions for contour rotation ###

def cart2pol(x, y):

    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def pol2cart(theta, rho):

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def rotate_contour(cnt, angle):

    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated


### merging functions ###

def find_metrics_data(item, root):

    df_name = ('{1}/{0}/{0}_data_metrics.csv').format(item, root)
    df = pd.read_csv(df_name)
    
    return df

def merge_metrics_data(directory_data_metrics, directory_save):
    
    dirlist = [item for item in os.listdir(directory_data_metrics) if os.path.isdir(os.path.join(directory_data_metrics, item))]

    df_export = []

    for item in dirlist:

        df_import = find_metrics_data(item, directory_data_metrics)

        if df_export is None:
            df_export.append(df_import)
        else:
            df_export.append(df_import)
    
    merged_data = pd.concat(df_export)

    merged_data.to_csv(('{}/data_metrics_merged.csv').format(directory_save), index=False)
    
def merge_metrics_and_metadata(directory_to_metrics, directory_to_metadata, directory_save):
    
    df_metrics = pd.read_csv(directory_to_metrics + '.csv')
    metadata = pd.read_csv(directory_to_metadata + '.csv')
    
    df_metrics['ka_cal_BP'] = metadata['ka_cal_BP']
    df_metrics['longitude'] = metadata['longitude']
    df_metrics['latitude'] = metadata['latitude']
    df_metrics['typology'] = metadata['typology']
    df_metrics['region'] = metadata['region']
    df_metrics['older_younger'] = metadata['older_younger']

    df_metrics.to_csv(('{}/data_metrics_merged_metadata.csv').format(directory_save), index=False)

# filter data set by .csv-data file
def filter_data(directory_to_data, directory_to_filter, directory_save):
    
    df_data = pd.read_csv(directory_to_data + '.csv')
    df_filter = pd.read_csv(directory_to_filter + '.csv')
    
    df_data['filter'] = df_filter['filter']
    
    df_filtered = df_data.loc[df_data['filter'] == 1]
    df_filtered = df_filtered.loc[:, df_filtered.columns != 'filter']
    
    df_filtered.to_csv(('{}/data_metrics_merged_metadata_filtered.csv').format(directory_save), index=False)
    
    return df_filtered

# group data set by grouping variable
def grouped_csvs(directory, directory_to_data, group_by):
    
    df = pd.read_csv(directory_to_data + '.csv')
    df_grouped = df.groupby(group_by)

    for k, group in df_grouped:
        # save each 'group' in a csv as follows
        group.to_csv('{}.csv'.format(directory + k), index = False)