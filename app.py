from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2
import numpy as np

APP_DIR=Path(__file__).resolve().parent
IMAGE_SIZE=224
LABEL_GOOD="GOOD"
LABEL_BAD="BAD"
DEFAULT_THRESHOLD=0.5
MODEL_TFLITE_PATH=APP_DIR/"model.tflite"
MODEL_KERAS_PATH=APP_DIR/"model.keras"
THRESHOLD_PATH=APP_DIR/"threshold.json"

def load_threshold():
    if THRESHOLD_PATH.exists():
        try:
            return float(json.loads(THRESHOLD_PATH.read_text(encoding="utf-8")).get("threshold",DEFAULT_THRESHOLD))
        except:
            pass
    return DEFAULT_THRESHOLD

def load_tflite():
    import tensorflow as tf
    if not MODEL_TFLITE_PATH.exists():
        return None,None,None
    interpreter=tf.lite.Interpreter(model_path=str(MODEL_TFLITE_PATH))
    interpreter.allocate_tensors()
    return interpreter,interpreter.get_input_details()[0],interpreter.get_output_details()[0]

def load_keras():
    if not MODEL_KERAS_PATH.exists():
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_KERAS_PATH,compile=False)

def predict_tflite(interpreter,input_info,output_info,crop):
    image=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE)).astype(np.float32)
    image=np.expand_dims(image,0)
    scale,zero=input_info["quantization"]
    if input_info["dtype"]==np.int8:
        image=np.round(image/scale+zero).clip(-128,127).astype(np.int8)
    elif input_info["dtype"]==np.uint8:
        image=np.round(image/scale+zero).clip(0,255).astype(np.uint8)
    else:
        image=image.astype(input_info["dtype"])
    interpreter.set_tensor(input_info["index"],image)
    interpreter.invoke()
    output=interpreter.get_tensor(output_info["index"])
    if output_info["dtype"] in (np.int8,np.uint8):
        scale,zero=output_info["quantization"]
        prob=(output.astype(np.float32)-zero)*scale
    else:
        prob=output.astype(np.float32)
    return float(np.squeeze(prob))

def predict_keras(model,crop):
    image=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE)).astype(np.float32)
    image=np.expand_dims(image,0)
    return float(np.squeeze(model.predict(image,verbose=0)))

def detect_oranges(frame):
    h,w=frame.shape[:2]
    scale=min(1.0,800/w)
    small=cv2.resize(frame,None,fx=scale,fy=scale) if scale<1 else frame.copy()

    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)

    # Vùng màu xanh -> vàng -> cam của quả.
    # H > 18 giúp loại phần lớn màu da.
    lower=np.array([18,80,45],np.uint8)
    upper=np.array([45,255,255],np.uint8)
    mask=cv2.inRange(hsv,lower,upper)

    kernel3=np.ones((3,3),np.uint8)
    kernel5=np.ones((5,5),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel3,iterations=1)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel5,iterations=2)
    mask=cv2.GaussianBlur(mask,(5,5),0)
    _,mask=cv2.threshold(mask,80,255,cv2.THRESH_BINARY)

    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    H,W=small.shape[:2]
    frame_area=H*W
    boxes=[]

    for c in contours:
        area=cv2.contourArea(c)
        if area<=0:
            continue

        ratio=area/frame_area
        if ratio<0.0015 or ratio>0.045:
            continue

        x,y,bw,bh=cv2.boundingRect(c)
        if bw<25 or bh<25:
            continue

        aspect=bw/max(bh,1)
        if not 0.60<=aspect<=1.45:
            continue

        perimeter=cv2.arcLength(c,True)
        if perimeter<=0:
            continue

        circularity=4*np.pi*area/(perimeter*perimeter)
        if circularity<0.55:
            continue

        hull=cv2.convexHull(c)
        hull_area=cv2.contourArea(hull)
        solidity=area/hull_area if hull_area>0 else 0
        if solidity<0.80:
            continue

        extent=area/(bw*bh)
        if extent<0.55:
            continue

        # Kiểm tra màu thực tế bên trong vùng quả.
        roi=hsv[y:y+bh,x:x+bw]
        color_mask=cv2.inRange(roi,lower,upper)
        color_ratio=cv2.countNonZero(color_mask)/max(bw*bh,1)
        if color_ratio<0.45:
            continue

        # Tâm và bán kính giúp loại các vùng hình dạng giống tay/cánh tay.
        (cx,cy),radius=cv2.minEnclosingCircle(c)
        circle_area=np.pi*radius*radius
        circle_ratio=area/circle_area if circle_area>0 else 0
        if circle_ratio<0.55:
            continue

        # Màu trung bình của vùng mask phải nghiêng về xanh/vàng/cam,
        # tránh vùng da.
        pixels=hsv[y:y+bh,x:x+bw][color_mask>0]
        if len(pixels)<20:
            continue
        mean_h=float(np.mean(pixels[:,0]))
        mean_s=float(np.mean(pixels[:,1]))
        if mean_h<20 or mean_h>43 or mean_s<80:
            continue

        pad=int(min(bw,bh)*0.12)
        x1=max(0,x-pad)
        y1=max(0,y-pad)
        x2=min(W,x+bw+pad)
        y2=min(H,y+bh+pad)

        boxes.append((int(x1/scale),int(y1/scale),int(x2/scale),int(y2/scale)))

    # NMS loại các khung trùng nhau.
    boxes=sorted(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]),reverse=True)
    kept=[]

    for box in boxes:
        x1,y1,x2,y2=box
        keep=True
        for k in kept:
            a1,b1,a2,b2=k
            ix=max(0,min(x2,a2)-max(x1,a1))
            iy=max(0,min(y2,b2)-max(y1,b1))
            inter=ix*iy
            area1=(x2-x1)*(y2-y1)
            area2=(a2-a1)*(b2-b1)
            union=area1+area2-inter
            if union>0 and inter/union>0.45:
                keep=False
                break
        if keep:
            kept.append(box)

    return sorted(kept,key=lambda b:(b[1],b[0]))

def classify(frame,boxes,threshold,interpreter,input_info,output_info,keras_model):
    results=[]

    for box in boxes:
        x1,y1,x2,y2=box
        crop=frame[y1:y2,x1:x2]

        if crop.size==0:
            continue

        if interpreter is not None:
            bad_prob=predict_tflite(interpreter,input_info,output_info,crop)
        else:
            bad_prob=predict_keras(keras_model,crop)

        bad_prob=float(np.clip(bad_prob,0,1))
        label=LABEL_BAD if bad_prob>=threshold else LABEL_GOOD
        confidence=bad_prob if label==LABEL_BAD else 1-bad_prob

        results.append((box,label,confidence,bad_prob))

    return results

def draw_result(frame,index,box,label,confidence):
    x1,y1,x2,y2=box

    if label==LABEL_GOOD:
        color=(0,255,0)
    else:
        color=(0,0,255)

    cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)

    text=f"Cam {index}: {label} {confidence*100:.1f}%"
    text_y=max(30,y1-10)

    cv2.putText(
        frame,text,(x1,text_y),
        cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2,cv2.LINE_AA
    )

def run_camera():
    threshold=load_threshold()

    interpreter,input_info,output_info=load_tflite()
    keras_model=None if interpreter is not None else load_keras()

    if interpreter is None and keras_model is None:
        raise FileNotFoundError("Không tìm thấy model.tflite hoặc model.keras.")

    if interpreter is not None:
        print("[OK] Đang dùng model.tflite INT8")
        print(f"[OK] Input: {input_info['dtype']}, scale={input_info['quantization'][0]}, zero={input_info['quantization'][1]}")
        print(f"[OK] Output: {output_info['dtype']}, scale={output_info['quantization'][0]}, zero={output_info['quantization'][1]}")

    print(f"[OK] Threshold: {threshold:.2f}")

    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

    if not cap.isOpened():
        cap=cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Không mở được camera.")

    print("[OK] Camera đang chạy. Nhấn Q để thoát.")

    while True:
        ok,frame=cap.read()

        if not ok:
            break

        boxes=detect_oranges(frame)

        results=classify(
            frame,boxes,threshold,
            interpreter,input_info,output_info,keras_model
        )

        # Vẽ từng quả.
        for i,(box,label,confidence,bad_prob) in enumerate(results,1):
            draw_result(frame,i,box,label,confidence)

        # Thông tin góc màn hình.
        cv2.putText(
            frame,
            f"So qua cam: {len(results)}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,(255,255,255),2,cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Threshold: {threshold:.2f}",
            (20,75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,(255,255,255),2,cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Nhan Q de thoat",
            (20,110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,(255,255,255),2,cv2.LINE_AA
        )

        cv2.imshow("Orange Quality - INT8",frame)

        key=cv2.waitKey(1)&0xFF

        if key==ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--camera",action="store_true")
    args=parser.parse_args()

    if args.camera:
        run_camera()
    else:
        print("Dùng: python app.py --camera")

if __name__=="__main__":
    main()