from paddleocr import PaddleOCR,draw_ocr
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore') 

ocr=PaddleOCR(lang='en')

def table_ocr(img_path):
    image_cv=cv2.imread(img_path)
    image_height=image_cv.shape[0]
    image_width=image_cv.shape[1]
    output=ocr.ocr(img_path)

    boxes=[out[0] for out in output[0] ]
    texts=[out[1][0] for out in output[0] ]
    probabilities=[out[1][1] for out in output[0]]

    image_boxes=image_cv.copy()
    for box,text in zip(boxes,texts):
        start_point=(int(box[0][0]),int(box[0][1]))
        end_point=(int(box[2][0]),int(box[2][1]))
        cv2.rectangle(image_boxes,start_point,end_point,color=(255,0,0),thickness=1)
        cv2.putText(img=image_boxes,text=text,org=start_point,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.33,color=(255,0,255),thickness=1)

    #cv2.imwrite(f"result{img_num}_00_01_detections.png",img=image_boxes)

    im=image_cv.copy()

    horizontal_boxes=[]
    vertical_boxes=[]
    for box in boxes:
        x_h,y_h,width_h,height_h=0,int(box[0][1]),image_width,int(box[2][1]-box[0][1])
        x_v,y_v,width_v,height_v=int(box[0][0]),0,int(box[2][0]-box[0][0]),image_height
        horizontal_boxes+=[[x_h,y_h,x_h+width_h,y_h+height_h]]
        vertical_boxes+=[[x_v,y_v,x_v+width_v,y_v+height_v]]
        cv2.rectangle(im,(x_h,y_h),(x_h+width_h,y_h+height_h),(255,0,0),1)
        cv2.rectangle(im,(x_v,y_v),(x_v+width_v,y_v+height_v),(255,0,0),1)

    #cv2.imwrite(f"result{img_num}_00_01_reconstruction.png",im)

    horizontal_output=tf.image.non_max_suppression(
        boxes=horizontal_boxes,
        scores=probabilities,
        max_output_size=1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )

    horizontal_lines=np.sort(np.array(horizontal_output))
    horizontal_lines

    im_non_max_suppression=image_cv.copy()

    for val in horizontal_lines:
        horizontal_box=horizontal_boxes[val]
        cv2.rectangle(im_non_max_suppression,(horizontal_box[0],horizontal_box[1]),(horizontal_box[2],horizontal_box[3]),(255,0,0),1)

    #cv2.imwrite(f"result{img_num}_00_01_reconstructed_horizontal_boxes.png",im_non_max_suppression)

    vertical_output=tf.image.non_max_suppression(
        boxes=vertical_boxes,
        scores=probabilities,
        max_output_size=1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )

    vertical_lines=np.sort(np.array(vertical_output))
    for val in vertical_lines:
        vertical_box=vertical_boxes[val]
        cv2.rectangle(im_non_max_suppression,(vertical_box[0],vertical_box[1]),(vertical_box[2],vertical_box[3]),(255,0,0),1)

    #cv2.imwrite(f"result{img_num}_00_01_reconstructed_boxes.png",im_non_max_suppression)

    output_array=[["" for i in range(len(vertical_lines))] for j in range(len(horizontal_lines))]
    np.array(output_array)

    vertical_unordered_boxes=[]
    for i in vertical_lines:
        vertical_unordered_boxes+=[vertical_boxes[i][0]]

    vertical_ordered_boxes=np.argsort(vertical_unordered_boxes)

    def intersection(box1,box2):
        return [box2[0],box1[1],box2[2],box1[3]]

    def iou(box1,box2):
        x1=max(box1[0],box2[0])
        y1=max(box1[1],box2[1])
        x2=min(box1[2],box2[2])
        y2=min(box1[3],box2[3])
        inter=abs(max((x2-x1),0)*max((y2-y1),0))
        if inter==0:
            return 0
        box1_area=abs((box1[2]-box1[0])*(box1[3]-box1[1]))
        box2_area=abs((box2[2]-box2[0])*(box2[3]-box2[1]))
        return inter/float(box1_area+box2_area-inter)

    for i in range(len(horizontal_lines)):
        for j in range(len(vertical_lines)):
            resultant=intersection(horizontal_boxes[horizontal_lines[i]],vertical_boxes[vertical_lines[vertical_ordered_boxes[j]]])
            
            for k in range(len(boxes)):
                box=boxes[k]
                the_box=[box[0][0],box[0][1],box[2][0],box[2][1]]
                if(iou(resultant,the_box)>0.1):
                    output_array[i][j]=texts[k]
    
    df=pd.DataFrame(data=output_array[1:],columns=output_array[0])
    if 'amount' in [x.lower() for x in df.columns]:
        amount_index=0
        for amount_index in range(len(df.columns)):
            if df.columns[amount_index].lower()=='amount':
                break
            amount_index+=1
        df=df.loc[:,[df.columns[1],df.columns[amount_index]]]
        df['file_name']=img_path.split('\\')[-1]
        df.rename(columns={df.columns[0]:'item_name',df.columns[1]:'item_amount'},inplace=True)
        df=df.loc[:,[df.columns[2],df.columns[0],df.columns[1]]]
        
    else:
        df=pd.DataFrame(data=None,columns=['file_name','item_name','item_amount'])
    #df.to_csv(f'table_extracted_csv_{img_num}.csv',index=False)
    return df
