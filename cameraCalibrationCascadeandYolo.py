import cv2
import numpy as np
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

#track yapılıp yapılmadığını tutan değişken
track=False

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

#trackbarlar için boş fonksiyon
def empty(a): pass

""" ---------------- Önceki haftadaki gereksiz fonksiyonlar ----------------
Rengi videoyu durdurarak seçim ile yapılan kodları ile Çok Hatalı pozitif sonuç verdiği için kullanamadığımız cascade ile ilgili kodlar
"""

# Renk Seçimi ile ilgili Trackbarları oluşturuyor
def createHSVTrackbar():
    cv2.namedWindow("HSV")
    cv2.resizeWindow("HSV", 640, 350)
    cv2.createTrackbar("hueMax", "HSV", 0, 255, empty)
    cv2.createTrackbar("hueLow", "HSV", 0, 255, empty)
    cv2.createTrackbar("satMax", "HSV", 0, 255, empty)
    cv2.createTrackbar("satLow", "HSV", 0, 255, empty)
    cv2.createTrackbar("valueMax", "HSV", 0, 255, empty)
    cv2.createTrackbar("valueLow", "HSV", 0, 255, empty)

    
# Renk Seçimi ile ilgili olan Trackbarlardaki değerleri alıyor
def getHSVTrackbar():
    # Trakbarlardan Hue, Saturation, ve Value ların max ve min değerlerini alıp değişkenlere atıyor
    hue = (cv2.getTrackbarPos("hueMax", "HSV"), cv2.getTrackbarPos("hueLow", "HSV"))
    saturation = (cv2.getTrackbarPos("satMax", "HSV"), cv2.getTrackbarPos("satLow", "HSV"))
    value = (cv2.getTrackbarPos("valueMax", "HSV"), cv2.getTrackbarPos("valueLow", "HSV"))
    # 0: max 1: Min
    color = ((hue[0], saturation[0], value[0]), (hue[1], saturation[1], value[1]))
    return color

#bulunan renk değerlerini Trackbara veriliyor
def setHSVTrackbar(color):
    cv2.setTrackbarPos("hueMax", "HSV", color[0][0])
    cv2.setTrackbarPos("hueLow", "HSV", color[1][0])
    cv2.setTrackbarPos("satMax", "HSV", color[0][1])
    cv2.setTrackbarPos("satLow", "HSV", color[1][1])
    cv2.setTrackbarPos("valueMax", "HSV", color[0][2])
    cv2.setTrackbarPos("valueLow", "HSV",color[1][2])

#önceki haftalarda kullandığımız isim yazmadan oluşan takip kutusu
def drawBox(img, bbox):
    global x,y
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (25, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
#önceki haftalardaki kullandığımız Renk ile takip Kodları
def drawFirstColor(orginalFrame,key):
    global track
     # hsvde çıkan bazı gürültüleri yok etmek için blur uygulanıyor
    blurred = cv2.GaussianBlur(orginalFrame, (15, 15), 0)
    blurredHsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    #c tuşuna basıldığı zaman orginal video adlı pencerede istediğimiz renk aralığına ait nesneyi seçiyoruz
    if key == ord("c"):
        bxo=cv2.selectROI("Orginal Video", orginalFrame, False)
        #renk aralığını bulup trackbarlara koyuyoruz
        setHSVTrackbar(colorPicker(blurredHsv,bxo))
        
    # trackbar dan gelen değerleri değişkenlere atıyoruz
    color = getHSVTrackbar()
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

        # a tuşuna basılınca takip algoritmasını başlatıyoruz ve etrafına bir kutu çiziyoruz
        bbox = []
        if key == ord("a"):
            bbox = cv2.boundingRect(c)
            drawBox(orginalFrame, bbox)
            track = True
            tracker.init(orginalFrame, bbox)
        # gelen değerler ile bir kutu şekli yapıyoruz
        box = cv2.boxPoints(rect)
        box = np.int64(box)

        # oluşan konturları görüntüye çiziyor
        cv2.drawContours(orginalFrame, [box], 0, (255, 0, 0), thickness=2)


# istenilen cascadenin dosya konumu verilir
folderName = "kirtasiye-cascade"
cascadeName = "kalem"
cascade = cv2.CascadeClassifier(folderName + "/" + cascadeName + ".xml")
#Cascade ile ilgili trackbarlar oluşturuluyor
def createCascadeTrackbar():
    cv2.namedWindow("Scale and Neighb/Box")
    cv2.resizeWindow("Scale and Neighb/Box", 640, 80)
    cv2.createTrackbar("Scale", "Scale and Neighb/Box", 400, 1000, empty)
    cv2.createTrackbar("Neighb/Box", "Scale and Neighb/Box", 4, 50, empty)
    
def drawCascade(frame):
    #trackbarlardan değerleri alıyoruz
    scaleVal = 1 + (cv2.getTrackbarPos("Scale", "Scale and Neighb/Box") / 1000)
    neighbor = cv2.getTrackbarPos("Neighb/Box", "Scale and Neighb/Box")
    global track
    if not track:
        # tespit edilen nesneyi değişkene atıyoruz
        nesne = cascade.detectMultiScale(frame, scaleVal, neighbor)
        centerRect = []
        # mouse noktası içerisinde olan nesne tespit kareleri
        RectN = []
        # tespit edilen nesnenin etrafına dikdörtgen çiziliyor
        for (casx, casy, casw, cash) in nesne:
            cv2.putText(frame, cascadeName, (casx, casy - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,(0, 255, 0))
            cv2.rectangle(frame, (casx, casy), (casx + casw, casy + cash), (0, 255, 0), 5)
            if Kontrol:
                if (pointInRect(Point, [casx, casy, casw, cash])):
                    # içerisinde ise o kutuyu bir değişkene atıyoruz ve orta noktasını da bir değişkene atıyoruz
                    RectN.append((casx, casy, casw, cash))
                    recx = int(np.round(casx + (casw / 2)))
                    recy = int(np.round(casy + (cash / 2)))
                    centerRect.append([recx, recy])
                # oluşan kutu dizisindeki orta noktaların mouse noktası ile uzaklığına bakıyoruz
        if len(centerRect)>0:
            distance = dist.cdist([Point], centerRect)
            # en yakın uzaklıktaki indexi alıyoruz ve o kutuyu değişkene atıyoruz
            distanceIndex = np.argmin(distance)
            RectN = RectN[distanceIndex]
            # bulunan kutunun etrafını çiziyoruz ve takip algoritmamızı başlatıyoruz
            drawBox(frame, RectN)
            track = True
            tracker.init(frame, RectN) 
            
""" ---------------- Önceki haftadaki gereksiz fonksiyonlar bitti ----------------"""

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
                    cv2.imshow('Kamera Kalibrasyonu', img)
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
                cv2.imshow('Kamera Kalibrasyonu', img)
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


#resimde  verilen box kadar boyutlu alanın içerisindeki max ve min hsv kodları bulunuyor
def colorPicker(frame,box):
    rx, ry, w, h = box[0],box[1],box[2],box[3]
    cropImg = frame[ry:ry + h, rx:rx + w]
    h, s, v = cv2.split(cropImg)
    color=((int(np.amax(h)+15),int(np.amax(s)+15),int(np.amax(v)+15)),(int(np.amin(h)-15),int(np.amin(s)-15),int(np.amin(v)-15)))
    return color

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

Point = None
Kontrol = False


#yolonun uyumluluk oranı, ismi ve boyutuna göre bir çerçeve hazırlar
def drawBoxforYolo(img, bbox, label, confidence, main):
    global x,y
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (25, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(img, main + " >> "+ label + " " + confidence, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


#kullanılabilecek trackerları tutan değişken
opencvTrackers = {"boosting": cv2.legacy.TrackerBoosting_create(),
                  "mil": cv2.legacy.TrackerMIL_create(),
                  "kcf": cv2.legacy.TrackerKCF_create(),
                  "tld": cv2.legacy.TrackerTLD_create(),
                  "medianflow": cv2.legacy.TrackerMedianFlow_create(),
                  "mosse": cv2.legacy.TrackerMOSSE_create(),
                  "csrt": cv2.legacy.TrackerCSRT_create()}
#kullanılan trackerı oluşturan değişken
trackerName = "csrt"
tracker = opencvTrackers[trackerName]


# mouse tıklamasını alan bir fonksiyon
def click_event(event, mouseX, mouseY, flags, params):
    global Point,Kontrol

    # mouse tıklama eventi  mouse x ve y koordinatlarını değişkenlere atıyor
    if event == cv2.EVENT_LBUTTONDOWN:       
        Kontrol = True
        Point = [mouseX, mouseY]

#kalibrasyon yapılıp fov değerleri bulunuyor
# fovx,fovy
fov = calibrateAndFov()
# gKuzeyX = int(input("Lütfen pusula yardımı ile kameranızın baktığı yönü derece olarak giriniz:"))
# gKuzeyY = int(input("Lütfen kameranızın baktığı yukarı-aşağı doğru baktığı yönü açı derecesi olarak giriniz:"))
gKuzeyX = 129
gKuzeyY = 10


cv2.namedWindow("Contour")

# Geçen haftaki kodların çalışmaması için kod
if False:
    # İstediğimiz renkleri aralığını seçmek için trackbar koyuyoruz
    createHSVTrackbar()
    # nesne tespiit ayarlaması yapmak için
    createCascadeTrackbar()

    cv2.createTrackbar("Color/Cascade/Yolo", "Contour", 2, 2, empty)

#yolonun kalitesini yükselten size değişkenini ayarlanması için bir trackbar
cv2.createTrackbar("YoloBox", "Contour", 12, 20, empty)

# işleyeceği videoyu alıyor
capture = cv2.VideoCapture(0)

while True:

    # videonun çalışıp çalışmadığı ve gelen kareleri değişkene atılıyor
    success, orginalFrame = capture.read()
    height, width, _ = orginalFrame.shape
    key = cv2.waitKey(1)
    
    #önceki haftalara ait kodların çalışmayıp sadece yolunun çalışması için
    choise=2
    
    #geçen haftaki kodlardan
    # trackbarlardaki bilgiyi alıyoruz
    #choise = cv2.getTrackbarPos("Color/Cascade/Yolo", "Contour")

    # eğer video oynuyorsa işlemleri yapıyor
    if success:


        #orta noktayı tutan değişken
        center = None
        # Seçilen renk aralığında nesnelerin tespitini ve takibini yapan fonksiyonu çağıran kod
        if choise == 0:
           drawFirstColor(orginalFrame,key)
        # Cascade ile nesnelerin tespitini ve takibini yapan fonksiyonu çağıran kod
        if choise == 1:
            drawCascade(orginalFrame)
        
        # yolo ile nesnelerin tespitini ,takibini ve sınıflandırılmasını yapan kod
        if choise == 2:
            #takip işlemi gerçekleşmiyorsa
            if not track:
                yoloBox=cv2.getTrackbarPos("YoloBox", "Contour")
                # görüntüden 4 boyutlu bir blob oluşturur. blob aynı yükseklik ve genişlikteki işlenmiş görüntü topluluğudur.
                blob = cv2.dnn.blobFromImage(orginalFrame, 1 / 255, (32+32*yoloBox, 32+32*yoloBox), (0, 0, 0), swapRB=True, crop=False)
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
                
                # uyumluluğu %50den fazla olan nesneler için oluşturulmuş index için isim, uyumluluk değişkenlere atılı, seçilen renk belirlenir ve dikdörtgeni çizilir
                if len(indexes) > 0:
                    for i in indexes.flatten():
                        yolox, yoloy, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        confidence = str(round(confidences[i], 2))
                        cv2.rectangle(orginalFrame, (yolox, yoloy), (yolox + w, yoloy + h), (0,255,0), 2)
                        #eğer mousela tıklanmışsa tıklanan alanda uyumlu nesne olup olmadığı kontrol edilir varsa takip başlatılır
                        if (Kontrol == True):
                            if (pointInRect(Point, boxes[i]) == True):
                                tracker.init(orginalFrame, boxes[i])
                                track = True
                                break
                #eğer mouse a tıklanmış ve track oluşmamışsa buraya giriliyor
                if Kontrol and not track:
                    #ilk önce renk ayrımını daha sağlıklı yapbilmek için blur atılıyor sonra HSV formatına çevirliyor
                    blurred = cv2.GaussianBlur(orginalFrame, (15, 15), 0)
                    blurredHsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                    
                    #tıklanılan nokta etrafında 10,10 pxlik bir kare yapılıyor ve o karedeki renklerin üst ve alt noktaları belirleniyor
                    baox= Point[0]-5,Point[1]-5,10,10
                    color=colorPicker(blurredHsv,baox)
                    
                    # Seçilen renk aralığında maske yapılıyor
                    mask = cv2.inRange(blurredHsv, color[1], color[0])
        
                    # maskedeki bazı kusurları düzeltilmesi için erosion ve dilation işlemleri yapılıyor
                    mask = cv2.erode(mask, None, iterations=4)
                    mask = cv2.dilate(mask, None, iterations=4)
        
                    # Maske görüntüsünde oluşan kenarlara göre kontur buluyor
                    (contours, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
                    # eğer kontur varsa içine girilir
                    if len(contours) > 0 and not track:
                        #konturları tek tek dolaşıyor
                        for cs in contours:
                            #konturlara bir dikdörgen çiziyor
                            rect=cv2.boundingRect(cs)
                            #eğer tıklanılan nokta kontur içerisinde ise tracker başlatılıyor
                            if pointInRect(Point,rect):
                                print("ok")
                                label,confidence="bilinmiyor","1.0"
                                drawBoxforYolo(orginalFrame, rect,label,confidence,main)
                                track = True
                                tracker.init(orginalFrame, rect)
                #hiçbir nesne bulunmamışsa tıklamayı kaldırıyor
                Kontrol=False
            
        if track:
            #tracker update ediliyor
            ret, bbox = tracker.update(orginalFrame)
            # eğer yolo çalışmıyorsa önceki haftalardaki gibi isim olmayan kutu çiziliyor
            if ret and choise!=2:
                drawBox(orginalFrame, bbox)
            elif ret:
            #takip başladığında nesnenin ismi ve uyumluluğu gösterilir
                drawBoxforYolo(orginalFrame, bbox, label, confidence, main)
            else:
                cv2.putText(orginalFrame, "Lost", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #orta noktayı bir değişkene atılır ve orta nokta çizilir
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

        # eğer track olmuşsa
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
            
            #x ve y koordinatlarının fov derece karşılığı bulunuyor
            uzaklikx = xc * xfov
            uzakliky = yc * yfov

            #Bulunan fov derece karşılıklarını gerçek kuzey ve gerçek güneylerine bakılıyor
            kuzeyx = gKuzeyX + uzaklikx
            kuzeyy = gKuzeyY + uzakliky

            s = "Kuzey: {}, yukselis: {}, ".format((kuzeyx), (kuzeyy))
            cv2.putText(orginalFrame, s, (25, 95), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)
            if (uzakliky < 0):
                cv2.putText(orginalFrame, "Alcaliyor", (25, 115), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255),2)
            if (uzakliky > 0):
                cv2.putText(orginalFrame, "Yukseliyor", (25, 115), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255),2)
        
        # oluşan konturlu görüntüyü gösteriliyor
        cv2.imshow("Contour", orginalFrame)
        #mouse eventi almak için click_event fonksiyonunu çağırıyor
        cv2.setMouseCallback("Contour", click_event)
        #bütün ekranları kapatıyor
    if key == ord("q"): break
capture.release()
cv2.destroyAllWindows()

