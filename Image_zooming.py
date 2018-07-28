import cv2
import argparse
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Path to input image", required=True)
ap.add_argument("-p", "--pivot-point", help="Pivot point coordinates x, y separated by comma (,)", required=True)
ap.add_argument("-s", "--scale", help="Scale to zoom", type=int, required=True)
args = vars(ap.parse_args())

image_path = args["image"]
x, y = map(int, args["pivot_point"].split(","))
scale = args["scale"]
image = cv2.imread(image_path)
#image = image.tolist()

#################################################################################################################
#My Code Starts here
#################################################################################################################



b = image
m,n,_ = b.shape

'''
Find the offset of the image, where it should be cropped.  
'''

row_delta = (m/scale)/2
column_delta = (n/scale)/2
x_offset_top = x - row_delta - 2
x_offset_bottom = x + row_delta + 2
y_offset_left = y - column_delta - 2
y_offset_right = y + column_delta + 2


'''
If the corners of the crop fall beyond the dimensions of the original image, push the pivot appropriately so that corner is covered. 
'''

if x_offset_top<0:
    x_offset_bottom += -1*(x_offset_top)
    x_offset_top = 0 
if x_offset_bottom>m:
    x_offset_top -= (x_offset_bottom-m)
    x_offset_bottom = m
if y_offset_left<0:
    y_offset_right += -1*(y_offset_left)
    y_offset_left = 0
if y_offset_right>n:
    y_offset_left -= (y_offset_right-n)
    y_offset_right = n


'''
Image is cropped, so that only pixels surrounding the pivot is included. Once scaled, the cropped image should be of same dimensions as original.
Calculations are localised to only those pixels which are required for scaling the image at the pivot.
'''
crop = b[x_offset_top:x_offset_bottom,y_offset_left:y_offset_right]
crop_m,crop_n,_ = crop.shape


'''
x,y will remain same as given pivot if no overflow happens. 
Else, the pivot is shifted to include the corners of the image, as explained above.
'''
x,y =  (x_offset_bottom+x_offset_top)/2,(y_offset_left+y_offset_right)/2



'''
K-TIMES ZOOMING ALGO
'''

out = np.zeros((crop_m,scale*(crop_n-1)+1,3),dtype=np.int16)
out[:,::scale] = crop

for row in range(0,out.shape[0]):
    for column in range(0,out.shape[1]-scale,scale):
        for channel in range(3):
            diff = out[row,column,channel] - out[row,column+scale,channel]
            if diff>0:
                op = diff//scale
                for mid_elem in range(0,scale-1):
                    out[row,column+mid_elem+1,channel] = out[row,column+mid_elem,channel] - op
            else:
                op = -1*diff//scale
                for mid_elem in range(0,scale-1):
                    out[row,column+mid_elem+1,channel] = out[row,column+mid_elem,channel] + op


out_final = np.zeros((scale*(crop_m-1)+1,scale*(crop_n-1)+1,3),dtype=np.int16)
out_final[::scale,:,:] = out

for column in range(0,out_final.shape[1]):
    for row in range(0,out_final.shape[0]-scale,scale):
        for channel in range(0,3):
            diff = out_final[row,column,channel] - out_final[row+scale,column,channel]
            if diff>0:
                op = diff//scale
                for mid_elem in range(0,scale-1):
                    out_final[row+mid_elem+1,column,channel] = out_final[row+mid_elem,column,channel] - op
            else:
                op = -1*diff//scale
                for mid_elem in range(0,scale-1):
                    out_final[row+mid_elem+1,column,channel] = out_final[row+mid_elem,column,channel] + op


'''
All co-ordinates whose value is equal to pixel values for given pivot.
'''
final_shape = out_final.shape
pivot_points = []
for i in range(out_final.shape[0]):
    for j in range(out_final.shape[1]):
        if out_final[i,j,0] == b[x,y,0] and out_final[i,j,1] == b[x,y,1] and out_final[i,j,2] == b[x,y,2]: 
            pivot_points.append((i,j))


'''
Find the co-ordinate of the value (equal to pivot pixel) closest to the center of the image. 
'''
image_center = (m/2,n/2)
min = 100000
for index in range(len(pivot_points)):
    pivot = pivot_points[index]
    diff_y,diff_x = pivot[0] - image_center[0],pivot[1] - image_center[1]
    diff_y,diff_x = diff_y*diff_y,diff_x*diff_x
    if (diff_x+diff_y)<min:
        min = diff_x+diff_y
        main_pivot = pivot_points[index]

'''
Crop the image so that resulting image size is same as original.
'''
if image_center!=main_pivot:
    x_offset = main_pivot[1] - image_center[1]
    y_offset = main_pivot[0] - image_center[0]
    if (final_shape[0] - m) == y_offset and (final_shape[1] - n) > x_offset:
        out_final = out_final[y_offset:,x_offset:-1*(final_shape[1]-n-x_offset)]    
    if (final_shape[1] - n) == x_offset and (final_shape[0] - m) > y_offset:
        out_final = out_final[y_offset:-1*(final_shape[0]-m-y_offset),x_offset:]
    if (final_shape[1] - n) > x_offset and (final_shape[0] - m) > y_offset:
        out_final = out_final[y_offset:-1*(final_shape[0]-m-y_offset),x_offset:-1*(final_shape[1]-n-x_offset)]


#################################################################################################################
#My Code Ends here
#################################################################################################################

cv2.imwrite("zoomed_image.jpg", np.array(out_final, dtype="uint8"))
