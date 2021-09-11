# Final
Proje kısaca kamera kalibrasyonu yapıldıktan sonra kameranın fov değeri bulunuyor. Ondan sonra kameraya önceden eğitilmiş yolo nesne tespiti algoritması sayesinde eğitilen nesneyi ve ona benzer nesneleri gösterince tespit ediliyor. Tespit edilen nesneye tıklanınca o nesneye tıklanınca nesnenin adı yazılıyor ve nesneyi takip ediliyor. Eğer nesneyi tespit edilememiş ise ama takip işlemi yapılması isteniyorsa gene o nesneye tıklanıyor ve o nesnedeki renkleri sayesinde nesnenin bilinmediğini söyleniyor ve o nesneyi takip işlemi başlatılıyor. Takip işlemi sırasında kullanıcının girdiği pusula yardımı ile kameranın baktığı kuzey derecesi sayesinde ve bulunan fov açısı ile x ve y koordinatlarını derece değerine çevrilip takip edilen nesnenin kameraya göre kuzey ve yükseliş açılarını yazılıyor.


## Projenin Çalıştırılması için Gerekenler
- Python
- Pyhton kodları çalıştırabilen bir ide 
- Opencv contrib
- Kamera

  #### Opencv Contrib Kütüphanesi Yüklemek
  - Anaconda Navigator ile Cmd veya powershell prompt açılır
  - Çıkan konsola ``` pip install opencv-contrib-python ``` yazılır ve enter a basılır pip otomatik olarak beatifulsoup kütüphanesini ekler
  - Numpy kütüphanesi yoksa gene konsola  ``` pip install numpy ``` yazılarak kütüphane yüklenir
  - Scipy kütüphanesi yoksa gene konsola  ``` pip install scipy ``` yazılarak kütüphane yüklenir

## Proje Nasıl Çalıştırılır
- Proje (``` cameraCalibrationCascadeandYolo.py ```) bir pyhton derleyecisi ile açılır ve çalıştırılır
- Program Çalıştırıldığı zaman konsola “Lütfen Bir Karenin Kenarının Değerini Milimetre Cinsinden Giriniz” gibi bir cümle yazıyorsa kamera kalibrasyonu yapılması gerekmektedir.
- Kalibrasyonu yapmak için ``` pattern.png ``` Yazıcıdan çıkartılır.
- Çıkartılan satranç şeklinin bir karesinin uzunluğu ölçülür
- Çıkarttığımız şekli sert bir zemin üzerine yapıştırılır
- Konsoldaki çıkan yazıya ölçtüğümüz uzunluk girilir
- Satranç şekli ile kameranın karşısına geçilir
- Çıkmak istediğimiz zaman q tuşuna basarak çıkabilirsiniz
- Karşımıza çıkan Kamera Kalibrasyonu adlı görüntüden bakarak kalan resim sayısı kadar olabildiğince farklı şekilde olacak şekilde satranç şekli kameraya gösterilir
- Eğer kalibrasyon işlemi yapılmış ise ve tekrar yapılınması isteniyor ise program kapalı bir konumda programın çalıştırıldığı klasördeki cameraMatrix.cam dosyası silinebilir/ adı değiştirilinebilir.
- Kameranın gerçek hayattaki pusula yardımı ile ölçülmüş şekilde baktığı yönün derecesi girilir
- Kameranın baktığı yere paralel olarak kaç derece eğim ile baktığının yükselti derecesi girilir 
- Kamera Kalibrasyonu bittiğinde Contour Adlı bir kamera görüntüsü çıkar
- Contour ekranını da istenildiği zaman q tuşuna basılarak program kapatılınabilir
- Nesne tespitinde istenilen hıza ve doğruluk derecesine ulaşılına kadar yoloBox Trackbarı ile oynanılanabilir
- Contour ekranındaki tespit edilen veya edinilmeyen takip edinilmesi istenilen cismin üzerine Mouse yardımı ile tıklanır.
- Nesne takip edinilebilmişse Contour ekranında Tracking ve altında o cismin o zamanki konumu ile ilgili değerler yazılır ve nesne etrafına pembe renkli bir kutu çizilir
  - Ortadan x ve y değerleri contour ekranındaki kırmızı çizgilerin kesişim noktası 0,0 olacak şekilde x ve y koordinatları
  - Kuzey ve Yükseliş açıları ise girilen kuzey ve yükseliş açılarından derece olarak orta noktadan uzaklığı verilir
  - Alçalıyor/ Yükseliyor yazısı ise Contour ekranının orta çizgisinden aşağıda mı yukarısında mı olduğu yazılıyor

# Proje Hakkında Video
Projenin Tanıtılması, Yolonun Kendi Verilerle Eğitilmesi ve Projenin Kodlarının İncelenmesi konularını içeren video https://youtu.be/nSAPh7fIqvI
