import cv2
import numpy as np
import time
from collections import deque
import math
import pickle
import os.path as os

# nesne merkezini depolayacak veri tipi
# kaç tane merkez noktası hatırlayacağı
buffer_size = 32
# pts bahsedilen merkezlerin noktalari
pts = deque(maxlen=buffer_size)


def empty(a): pass

#istenilen cascadenin dosya konumu verilir
cascadeName = "bardak"
cascade = cv2.CascadeClassifier(cascadeName + ".xml")

# Trackbarları oluşturuyor
def createHSVTrackbar():
    cv2.namedWindow("HSV")
    cv2.resizeWindow("HSV", 650, 350)
    cv2.createTrackbar("hueMax", "HSV", 0, 255, empty)
    cv2.createTrackbar("hueLow", "HSV", 0, 255, empty)
    cv2.createTrackbar("satMax", "HSV", 0, 255, empty)
    cv2.createTrackbar("satLow", "HSV", 0, 255, empty)
    cv2.createTrackbar("valueMax", "HSV", 0, 255, empty)
    cv2.createTrackbar("valueLow", "HSV", 0, 255, empty)


def createTrackbar():
    cv2.namedWindow("Scale and Neighb/Box")
    cv2.resizeWindow("Scale and Neighb/Box", 650, 350)
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

x = 0
y = 0

def drawBox(img, bbox):
    global x
    global y
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (25, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


# videolar arası geçiş yaparken kodları bir daha yapmasın diye fonksiyon içerisinde
def videos(video):
    # İstediğimiz renkleri aralığını seçmek için trackbar koyuyoruz
    createHSVTrackbar()
    # nesne tespiit ayarlaması yapmak için
    createTrackbar()

    # işleyeceği videoyu alıyor
    capture = cv2.VideoCapture(0)
    opencvTrackers = {"boosting": cv2.legacy.TrackerBoosting_create(),
                      "mil": cv2.legacy.TrackerMIL_create(),
                      "kcf": cv2.legacy.TrackerKCF_create(),
                      "tld": cv2.legacy.TrackerTLD_create(),
                      "medianflow": cv2.legacy.TrackerMedianFlow_create(),
                      "mosse": cv2.legacy.TrackerMOSSE_create(),
                      "csrt": cv2.legacy.TrackerCSRT_create()}
    trackerName = "kcf"
    tracker = opencvTrackers[trackerName]
    track = False

    while True:

        # videonun çalışıp çalışmadığı ve gelen kareleri değişkene atılıyor
        success, orginalFrame = capture.read()

        key = cv2.waitKey(1)
        # trackbar dan gelen değerleri değişkenlere atıyoruz
        color = getHSVTrackbar()

        #trackbarlardaki bilgiyi alıyoruz
        scaleVal = 1 + (cv2.getTrackbarPos("Scale", "Scale and Neighb/Box") / 1000)
        neighbor = cv2.getTrackbarPos("Neighb/Box", "Scale and Neighb/Box")

        # eğer video oynuyorsa işlemleri yapıyor
        if success:

            # videoyu daha yavaş bir şekilde oynatılması için bekleniyor ve video oynatılıyor
            time.sleep(0.01)
            cv2.imshow("Orginal Video", orginalFrame)

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
            center = None

            #tespit edilen nesneyi değişkene atıyoruz
            nesne = cascade.detectMultiScale(orginalFrame, scaleVal, neighbor)
            #tespit edilen nesnenin etrafına dikdörtgen çiziliyor
            for (x, y, w, h) in nesne:
                cv2.putText(orginalFrame, cascadeName, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0))
                cv2.rectangle(orginalFrame, (x, y), (x + w, y + h), (0, 255, 0), 5)

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

            if track:
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
            if len(contours) > 0 or track:
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

            # oluşan konturlu görüntüyü gösteriliyor
            cv2.imshow("Contour", orginalFrame)

        if key == ord("q"): break
    capture.release()
    cv2.destroyAllWindows()


# fovx,fovy
fov = calibrateAndFov()
gKuzeyX = int(input("Lütfen pusula yardımı ile kameranızın baktığı yönü derece olarak giriniz:"))
gKuzeyY = int(input("Lütfen kameranızın baktığı yukarı-aşağı doğru baktığı yönü açı derecesi olarak giriniz:"))

videos(0)



