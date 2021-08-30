import cv2
import numpy as np
import time
from collections import deque
import math
import pickle
import os.path as os
import scipy.spatial.distance as dist

# nesne merkezini depolayacak veri tipi
# kaç tane merkez noktası hatırlayacağı
buffer_size = 32
# pts bahsedilen merkezlerin noktalari
pts = deque(maxlen=buffer_size)

#yolo klasör ve dosya isimleri
yoloFolderName = "kirtasiye-yolo/"
main = yoloFolderName.split("-")
main = main[0]
yoloModel = "yolo4.weights"
yoloConfig = "yolo4.cfg"
yoloObjectNames = "object.names"

#yolonun tespit ettiği nesnelerin isimleri isimlerinin olduğu dosyadan okunarak classes arrayine atılır
classes = []
with open(yoloFolderName + yoloObjectNames, "r") as f:
    classes = f.read().splitlines()

# dnn deep neural network modülüdür.
#eğitilmiş bir modelin config dosyası ve weight dosyası verilerek opencvnin algılaması sağlanır
net = cv2.dnn.readNet(yoloFolderName + yoloModel, yoloFolderName + yoloConfig)


def empty(a): pass


# istenilen cascadenin dosya konumu verilir
folderName = "kirtasiye-cascade"
cascadeName = "kalem"
cascade = cv2.CascadeClassifier(folderName + "/" + cascadeName + ".xml")


# Trackbarları oluşturuyor
def createHSVTrackbar():
    cv2.namedWindow("HSV")
    cv2.resizeWindow("HSV", 640, 350)
    cv2.createTrackbar("hueMax", "HSV", 0, 255, empty)
    cv2.createTrackbar("hueLow", "HSV", 0, 255, empty)
    cv2.createTrackbar("satMax", "HSV", 0, 255, empty)
    cv2.createTrackbar("satLow", "HSV", 0, 255, empty)
    cv2.createTrackbar("valueMax", "HSV", 0, 255, empty)
    cv2.createTrackbar("valueLow", "HSV", 0, 255, empty)


def createTrackbar():
    cv2.namedWindow("Scale and Neighb/Box")
    cv2.resizeWindow("Scale and Neighb/Box", 640, 80)
    cv2.createTrackbar("Scale", "Scale and Neighb/Box", 400, 1000, empty)
    cv2.createTrackbar("Neighb/Box", "Scale and Neighb/Box", 4, 50, empty)


# Trackbarlardaki değerleri alıyor
def getHSVTrackbar():
    # Trakbarlardan Hue, Saturation, ve Value ların max ve min değerlerini alıp değişkenlere atıyor
    hue = (cv2.getTrackbarPos("hueMax", "HSV"), cv2.getTrackbarPos("hueLow", "HSV"))
    saturation = (cv2.getTrackbarPos("satMax", "HSV"), cv2.getTrackbarPos("satLow", "HSV"))
    value = (cv2.getTrackbarPos("valueMax", "HSV"), cv2.getTrackbarPos("valueLow", "HSV"))
    # 0: max 1: Min
    color = ((hue[0], saturation[0], value[0]), (hue[1], saturation[1], value[1]))
    return color


# Seçilen alandaki bölgeyi kırpıyor ve max ve min değerleri bulup trackbarların değerlerini güncelliyor
def colorPicker(frame, hsvFrame):
    cropCords = cv2.selectROI("Orginal Video", frame, False)
    x, y, w, h = int(cropCords[0]), int(cropCords[1]), int(cropCords[2]), int(cropCords[3])
    cropImg = hsvFrame[y:y + h, x:x + w]
    h, s, v = cv2.split(cropImg)
    cv2.setTrackbarPos("hueMax", "HSV", np.amax(h))
    cv2.setTrackbarPos("hueLow", "HSV", np.amin(h))
    cv2.setTrackbarPos("satMax", "HSV", np.amax(s))
    cv2.setTrackbarPos("satLow", "HSV", np.amin(s))
    cv2.setTrackbarPos("valueMax", "HSV", np.amax(v))
    cv2.setTrackbarPos("valueLow", "HSV", np.amin(v))


# Kameranın fov derecelerini bulmak için fonksiyon
def calibrateAndFov():
    # önceden kameranın kalibrasyonu yapılıp matrix değerleri alımışsa onları yüklüyor
    if os.exists("cameraMatrix.cam"):
        with open('cameraMatrix.cam', 'rb') as filehandle:
            cameraMatrix = pickle.load(filehandle)
        filehandle.close()

    # yoksa kalibrasyon işlemini başlatıyor
    else:
        # Satranç karesinin gerçek hayattaki boyutunu veriyoruz
        mm = int(input("Lütfen Bir Karenin Kenarının Değerini Milimetre Cinsinden Giriniz:"))

        # Kullanacağımız satranç tahtasında kaç tane kare olduğunu söylüyoruz
        patternSize = (9, 6)

        # 3 boyutlu noktalar için world koordinatları tanımlıyoruz
        patternPoints = np.zeros((np.prod(patternSize), 3), np.float32)
        patternPoints[:, :2] = np.indices(patternSize).T.reshape(-1, 2)

        # Satranç tahtasındaki her karenin 3d düzlemde noktalarını tutacak bir vektor oluşturuyoruz
        objPoints = []
        # Satranç tahtasındaki her karenin 2d düzlemde noktalarını tutacak bir vektor oluşturuyoruz
        imgPoints = []

        capture = cv2.VideoCapture(0)

        # Satranç tahtasında köşeleri bulunduktan sonra daha çok düzeltme
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, mm, 0.1)

        # Kalibrede kullanacağımız fotoğraf sayısını belirliyoruz ve iki tane sayaç için değişken oluşturuyoruz
        imgGoal = 30
        imgCount = 0
        frameCount = -1

        while True:
            calSuccess, img = capture.read()
            if calSuccess:
                # frame sayısını yazdırılıyor
                frameCount = frameCount + 1;
                s = "Frame:{}".format(frameCount)

                # durdurmak için tuş bekliyort
                key = cv2.waitKey(1)
                if key == ord('q'):
                    capture.release()
                    cv2.destroyAllWindows()
                    raise SystemExit
                # eger frame sayısı 20 nin katı değilse calibrasyon için fotoraf almıyor
                if not (frameCount % 20) == 0:
                    s = "{}kalan resim {}".format(s, str(imgGoal - imgCount))
                    cv2.putText(img, s, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)
                    cv2.imshow('Video', img)
                    continue

                # kalibrasyon için resmi gri tonlamalı siyah beyaza çeviriyor
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # resmin boyutlarını alıyor
                h, w = gray.shape[:2]

                # satranç karolarının köşelerini arıyor
                found, corners = cv2.findChessboardCorners(gray, patternSize, flags=cv2.CALIB_CB_FILTER_QUADS)

                # bulamazsa sonraki kareye geçmek için aşağıdaki satırları boş verip while döngüsüne devam ettiriyor
                if not found:
                    continue

                # Bulursa resim sayısını bir artıryor ve resim sayısı istenilen düzeyde ise döngüyü durduruyor
                imgCount = imgCount + 1
                if imgGoal == imgCount:
                    break

                # Satranç karolarının köşelerin olduğu yeri subPixel boyutunda düzeltme yapıyor
                cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

                # bulduğu köşeleri çiziyor
                cv2.drawChessboardCorners(img, patternSize, corners, found)

                # bulduğu koşe noktalarını vektörlere atıyor
                imgPoints.append(corners.reshape(1, -1, 2))
                objPoints.append(patternPoints.reshape(1, -1, 3))

                # görüntüyü gösteriyor ve 100 ms bekliyor
                cv2.imshow('Video', img)
                cv2.waitKey(100)

        # kalibrasyon fonksiyonunu çağırıp  kamera matrisini alıyor
        _, cameraMatrix, _, _, _ = cv2.calibrateCamera(objPoints, imgPoints, (w, h), None, None)

        # kamera matrisinde başka bir zaman kullanmak için görüntü boyutların boş yere koyuyor
        cameraMatrix[0, 1] = w
        cameraMatrix[0, 2] = h

        # kamera matrisini bir dahaya kullanmak için bir dosyaya atıyor
        with open("cameraMatrix.cam", "wb") as fileName:
            pickle.dump(cameraMatrix, fileName)
            pickle.dump((w, h), fileName)
        fileName.close()

        capture.release()
        cv2.destroyAllWindows()
    # Kamera matrisinden fov hesabı yapıyor
    # fov=(fovx,fovy)
    fov = (math.degrees(2 * (math.atan2(cameraMatrix[0, 1], 2 * cameraMatrix[0, 0]))),
           math.degrees(2 * (math.atan2(cameraMatrix[0, 2], 2 * cameraMatrix[1, 1]))))
    return fov

#bir noktanın bir dikdörtgen içinde olup olmadığını bulur
def pointInRect(point,rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False

x = 0
y = 0


def drawBox(img, bbox):
    global x
    global y
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (25, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

label = None
confidence = None
Point = None
Kontrol = False
Kontrol2 = False
boxx2 = None

Kontrol3 = False

#yolonun uyumluluk oranı, ismi ve boyutuna göre bir çerçeve hazırlar
def drawBoxforYolo(img, bbox, label, confidence, main):
    global x
    global y
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (25, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(img, main + " >> "+ label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

key = -1
img = None
nesne = None
opencvTrackers = {"boosting": cv2.legacy.TrackerBoosting_create(),
                  "mil": cv2.legacy.TrackerMIL_create(),
                  "kcf": cv2.legacy.TrackerKCF_create(),
                  "tld": cv2.legacy.TrackerTLD_create(),
                  "medianflow": cv2.legacy.TrackerMedianFlow_create(),
                  "mosse": cv2.legacy.TrackerMOSSE_create(),
                  "csrt": cv2.legacy.TrackerCSRT_create()}
trackerName = "kcf"
tracker = opencvTrackers[trackerName]

choise = None

# mouse tıklamasını alan bir fonksiyon
def click_event(event, mouseX, mouseY, flags, params):
    global track, tracker, nesne, Point,Kontrol

    # mouse tıklama eventi ve t tuşuna basılmışsa mouse x ve y koordinatlarını değişkenlere atayıp contour üzerine nokta bırakıyor
    if event == cv2.EVENT_LBUTTONDOWN:
        Kontrol = True
        Point = [mouseX, mouseY]
        if choise != 2:
            if len(nesne) > 0:
                centerRect = []
                # mouse noktasını tutan değişken
                mouseCenter = []
                # mouse noktası içerisinde olan nesne tespit kareleri
                RectN = []
                # cdist algoritmasını kullanabilmemiz için mouse noktalarını bir listeye atıyoruz
                mouseCenter.append((mouseX, mouseY))
                # tespit edilen kutuları tek tek dolaşıp mouse noktasının içerisinde olup olmadığını bakıyoruz
                for (casx, casy, casw, cash) in nesne:
                    if ((casx <= mouseX) and (casx + casw >= mouseX) and ((casy <= mouseY) and (casy + cash >= mouseY))):
                        # içerisinde ise o kutuyu bir değişkene atıyoruz ve orta noktasını da bir değişkene atıyoruz
                        RectN.append((casx, casy, casw, cash))
                        recx = int(np.round(casx + (casw / 2)))
                        recy = int(np.round(casy + (cash / 2)))
                        centerRect.append((recx, recy))
                        # oluşan kutu dizisindeki orta noktaların mouse noktası ile uzaklığına bakıyoruz
                distance = dist.cdist(mouseCenter, centerRect)
                # en yakın uzaklıktaki indexi alıyoruz ve o kutuyu değişkene atıyoruz
                distanceIndex = np.argmin(distance)
                RectN = RectN[distanceIndex]
                # bulunan kutunun etrafını çiziyoruz ve takip algoritmamızı başlatıyoruz
                drawBox(img, RectN)
                track = True
                tracker.init(img, RectN)

font = None

# videolar arası geçiş yaparken kodları bir daha yapmasın diye fonksiyon içerisinde
def videos(video):
    global key, img, tracker, track, nesne, label, confidence, boxx2, font
    # İstediğimiz renkleri aralığını seçmek için trackbar koyuyoruz
    createHSVTrackbar()
    # nesne tespiit ayarlaması yapmak için
    createTrackbar()

    cv2.namedWindow("Contour")
    cv2.createTrackbar("Color/Cascade/Yolo", "Contour", 2, 2, empty)

    # işleyeceği videoyu alıyor
    capture = cv2.VideoCapture(video)

    track = False

    while True:

        # videonun çalışıp çalışmadığı ve gelen kareleri değişkene atılıyor
        success, orginalFrame = capture.read()
        img = orginalFrame
        height, width, _ = img.shape
        key = cv2.waitKey(1)

        # trackbar dan gelen değerleri değişkenlere atıyoruz
        color = getHSVTrackbar()

        # trackbarlardaki bilgiyi alıyoruz
        scaleVal = 1 + (cv2.getTrackbarPos("Scale", "Scale and Neighb/Box") / 1000)
        neighbor = cv2.getTrackbarPos("Neighb/Box", "Scale and Neighb/Box")
        choise = cv2.getTrackbarPos("Color/Cascade/Yolo", "Contour")

        # eğer video oynuyorsa işlemleri yapıyor
        if success:

            # videoyu daha yavaş bir şekilde oynatılması için bekleniyor ve video oynatılıyor
            time.sleep(0.01)
            cv2.imshow("Orginal Video", orginalFrame)

            center = None
            if choise == 0:
                # hsvde çıkan bazı gürültüleri yok etmek için blur uygulanıyor
                blurred = cv2.GaussianBlur(orginalFrame, (15, 15), 0)
                blurredHsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

                if key == ord("c"):
                    colorPicker(orginalFrame, blurredHsv)

                # Seçilen renk aralığında maske yapılıyor
                mask = cv2.inRange(blurredHsv, color[1], color[0])

                # maskedeki bazı kusurları düzeltilmesi için erosion ve dilation işlemleri yapılıyor
                mask = cv2.erode(mask, None, iterations=4)
                mask = cv2.dilate(mask, None, iterations=4)
                # cv2.imshow("erode + dilate mask",mask)

                # Maske görüntüsünde oluşan kenarlara göre kontur buluyor
                (contours, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # eğer kontur varsa içine girilir
                if len(contours) > 0 and not track:
                    # ekrandaki en büyük konturu c değişkenine atılıyor
                    c = max(contours, key=cv2.contourArea)

                    # konturdan gelen görüntüyü oluşturabileceği en küçük dikdörtgeni özelliklerini rect değişkenine atıyor
                    rect = cv2.minAreaRect(c)

                    # gelen değerler ile bir kutu şekli yapıyoriz
                    bbox = []

                    if key == ord("a"):
                        bbox = cv2.boundingRect(c)
                        drawBox(orginalFrame, bbox)
                        track = True
                        tracker.init(orginalFrame, bbox)

                    box = cv2.boxPoints(rect)
                    box = np.int64(box)

                    # oluşan konturları görüntüye çiziyor
                    cv2.drawContours(orginalFrame, [box], 0, (255, 0, 0), thickness=2)

                    # Konturdaki orta noktayı buluyoruz ve bir nokta çiziyoruz
                    center = (int(np.round(rect[0][0])), int(np.round(rect[0][1])))
                    cv2.circle(orginalFrame, center, 5, (255, 0, 255), -1)
                    cv2.putText(orginalFrame, "Lost", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if choise == 1:
                if not track:
                    # tespit edilen nesneyi değişkene atıyoruz
                    nesne = cascade.detectMultiScale(orginalFrame, scaleVal, neighbor)

                    # tespit edilen nesnenin etrafına dikdörtgen çiziliyor
                    for (casx, casy, casw, cash) in nesne:
                        cv2.putText(orginalFrame, cascadeName, (casx, casy - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                                    (0, 255, 0))
                        cv2.rectangle(orginalFrame, (casx, casy), (casx + casw, casy + cash), (0, 255, 0), 5)
                    cv2.imshow("Contour", orginalFrame)

                # tuşa basıldığı zaman tespit edilen nesne varsa tıklanan en yakın nesneyi takip etmesi için
                """
                if key==ord("t"):
                    #mouse tıklama eventi başlatıyoruz
                    cv2.setMouseCallback("Contour", click_event)
                    cv2.waitKey(0)
                    #Nesne tespit edilmişse giriyoruz
                    if(len(nesne)>0):
                        #orta noktaları tutan değişken
                        centerRect=[]
                        #mouse noktasını tutan değişken
                        mouseCenter=[]
                        #mouse noktası içerisinde olan nesne tespit kareleri
                        RectN=[]
                        #cdist algoritmasını kullanabilmemiz için mouse noktalarını bir listeye atıyoruz
                        mouseCenter.append((mouseX,mouseY))
                        #tespit edilen kutuları tek tek dolaşıp mouse noktasının içerisinde olup olmadığını bakıyoruz
                        for (casx,casy,casw,cash) in nesne:
                            if((casx<=mouseX)and(casx+casw>=mouseX)and ((casy<=mouseY)and(casy+cash>=mouseY))):
                                #içerisinde ise o kutuyu bir değişkene atıyoruz ve orta noktasını da bir değişkene atıyoruz
                                RectN.append((casx,casy,casw,cash))
                                recx=int(np.round(casx+(casw/2)))
                                recy=int(np.round(casy+(cash/2)))
                                centerRect.append((recx,recy))         
                        #oluşan kutu dizisindeki orta noktaların mouse noktası ile uzaklığına bakıyoruz
                        distance=dist.cdist(mouseCenter,centerRect)
                        #en yakın uzaklıktaki indexi alıyoruz ve o kutuyu değişkene atıyoruz
                        distanceIndex=np.argmin(distance)
                        RectN=RectN[distanceIndex]
                        #bulunan kutunun etrafını çiziyoruz ve takip algoritmamızı başlatıyoruz
                        drawBox(orginalFrame,RectN)
                        track=True
                        tracker.init(orginalFrame, RectN)
                """
            if choise == 2:
                if not track:
                    # görüntüden 4 boyutlu bir blob oluşturur. blob aynı yükseklik ve genişlikteki işlenmiş görüntü topluluğudur.
                    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
                    #bloobları networke input olarak verilmesini sağlar
                    net.setInput(blob)
    
                    #çıktıların tabakalarını ayarlar
                    output_layers_names = net.getUnconnectedOutLayersNames()
                    layerOutputs = net.forward(output_layers_names)
    
                    boxes = []
                    confidences = []
                    class_ids = []

                    for output in layerOutputs:
                        for detection in output:
                            #tespit edilen neslerin uyumluluk yüzdelerini ve isimlerini kaydetmek için array oluşturulur
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            #uyumluluk %50den fazlaysa nesnenin çerçevesi boxes arrayine atılır, ismi ve uyumluluğu arraye atılır
                            if confidence > 0.5:
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)

                                yolox = int(center_x - w / 2)
                                yoloy = int(center_y - h / 2)

                                boxes.append([yolox, yoloy, w, h])
                                confidences.append((float(confidence)))
                                class_ids.append(class_id)

                    #maksimum olmayan noktalar belirlenir
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    #isim için yazı fontu ve belirlenen nesneler için rastgele renkler seçilir
                    font = cv2.FONT_HERSHEY_PLAIN
                    # uyumluluğu %50den fazla olan nesneler için oluşturulmuş index için isim, uyumluluk değişkenlere atılı, seçilen renk belirlenir ve dikdörtgeni çizilir
                    if len(indexes) > 0:
                        for i in indexes.flatten():
                            yolox, yoloy, w, h = boxes[i]
                            label = str(classes[class_ids[i]])
                            confidence = str(round(confidences[i], 2))
                            cv2.rectangle(img, (yolox, yoloy), (yolox + w, yoloy + h), (0,255,0), 2)
                            #eğer mousela tıklanmışsa tıklanan alanda uyumlu nesne olup olmadığı kontrol edilir varsa takip başlatılır
                            if (Kontrol == True):
                                if (pointInRect(Point, boxes[i]) == True):
                                    boxx2 = boxes[i]
                                    tracker.init(img, boxx2)
                                    track = True
                                    break

                if track:
                    ret, bbox = tracker.update(img)
                    #takip başladığında nesnenin ismi ve uyumluluğu gösterilir
                    if (ret):
                        drawBoxforYolo(img, bbox, label, confidence, main)
                    else:
                        cv2.putText(img, "Lost", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    center = (int(np.round(x + (bbox[2] / 2))), int(np.round(y + (bbox[3] / 2))))
                    cv2.circle(orginalFrame, center, 5, (255, 0, 255), -1)

            if track and choise != 2:
                ret, bbox = tracker.update(orginalFrame)
                if ret:
                    drawBox(orginalFrame, bbox)
                else:
                    cv2.putText(orginalFrame, "Lost", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                center = (int(np.round(x + (bbox[2] / 2))), int(np.round(y + (bbox[3] / 2))))
                cv2.circle(orginalFrame, center, 5, (255, 0, 255), -1)

            # Orta noktaları arkasında bir çizgi şeklinde iz bırakması için bir deque atılıyor
            pts.appendleft(center)

            # İz bırakması için şimdiki orta noktadan 32 frame önceki noktaya kadar olan noktalar arası çizgi çiziliyor
            for i in range(1, len(pts)):
                # eğer bir önceki veya şuanki nokta yoksa birşey yapılmıyor
                if pts[i - 1] is None or pts[i] is None: continue
                cv2.line(orginalFrame, pts[i - 1], pts[i], (0, 255, 255), 1)

                # orta noktayı bulmak için görüntünün boyutunu alınıyor ve 2 ye bölünüyor ve 3 piksellik bir çizgi çiziliyor
            height, width, _ = orginalFrame.shape
            x0 = int(width / 2)
            y0 = int(height / 2)
            cv2.line(orginalFrame, (int(width / 2), 0), (int(width / 2), height), (0, 0, 255), 3)
            cv2.line(orginalFrame, (0, int(height / 2)), (width, int(height / 2)), (0, 0, 255), 3)

            # orta noktadan her 50 piksel mesafe arası bir yeşil çizgi çiziliyor
            ss = 0
            for i in range(x0 + 50, width, 50):
                cv2.line(orginalFrame, (i, 0), (i, height), (0, 255, 0), 1)
                ss = ss + 1
                cv2.line(orginalFrame, (i - (ss * 50 * 2), 0), (i - (ss * 50 * 2), height), (0, 255, 0), 1)
            ss = 0
            for i in range(y0 + 50, height, 50):
                cv2.line(orginalFrame, (0, i), (width, i), (0, 255, 0), 1)
                ss = ss + 1
                cv2.line(orginalFrame, (0, i - (ss * 50 * 2)), (width, i - (ss * 50 * 2)), (0, 255, 0), 1)

            # eğer kontur veya track
            if track:
                # nesnenin merkezinin orta noktadan uzaklığı çıkarılıyor
                xc = int(center[0]) - x0
                yc = y0 - int(center[1])
                s = "Ortadan: x: {}, y: {}, ".format(xc, yc)
                cv2.putText(orginalFrame, s, (25, 65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)

                # orta noktadan cismin orta noktasına bir çizgi çiziliyor
                cv2.line(orginalFrame, (x0, y0), (int(np.round(center[0])), int(np.round(center[1]))), (255, 255, 0), 2)

                # 1px fov karşılığı
                xfov = fov[0] / width
                yfov = fov[1] / height

                uzaklikx = xc * xfov
                uzakliky = yc * yfov

                kuzeyx = gKuzeyX + uzaklikx
                kuzeyy = gKuzeyY + uzakliky

                s = "Kuzey: {}, yukselis: {}, ".format((kuzeyx), (kuzeyy))
                cv2.putText(orginalFrame, s, (25, 95), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)
                if (uzakliky < 0):
                    cv2.putText(orginalFrame, "Alcaliyor", (25, 115), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255),
                                2)
                if (uzakliky > 0):
                    cv2.putText(orginalFrame, "Yukseliyor", (25, 115), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255),
                                2)
            cv2.setMouseCallback("Contour", click_event)
            # oluşan konturlu görüntüyü gösteriliyor
            cv2.imshow("Contour", orginalFrame)

        if key == ord("q"): break
    capture.release()
    cv2.destroyAllWindows()


# fovx,fovy
fov = calibrateAndFov()
# gKuzeyX = int(input("Lütfen pusula yardımı ile kameranızın baktığı yönü derece olarak giriniz:"))
# gKuzeyY = int(input("Lütfen kameranızın baktığı yukarı-aşağı doğru baktığı yönü açı derecesi olarak giriniz:"))
gKuzeyX = 129
gKuzeyY = 0
videos(0)


