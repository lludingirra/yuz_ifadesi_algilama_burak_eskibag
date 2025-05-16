# Gerçek Zamanlı Duygu Tanıma Sistemi

Bu proje, kamera görüntüsünden alınan yüz verilerini kullanarak gerçek zamanlı duygu tanıma yapmayı amaçlamaktadır. MediaPipe kütüphanesi ile yüz işaret noktaları (face landmarks) tespit edilir ve makine öğrenimi modelleri kullanılarak duygular sınıflandırılır.

## Proje İçeriği

Bu depo aşağıdaki dosyaları içermektedir:

* **egitim.py:** Makine öğrenimi modelini eğitmek için kullanılan Python betiği.
* **yuz\_algila.py:** Kameradan yüz verilerini alarak işaret noktalarını tespit eden ve veri setine kaydeden Python betiği.
* **yuz\_algila\_test.py:** Eğitilmiş modeli kullanarak gerçek zamanlı duygu tahmini yapan Python betiği.
* **model.pkl:** Eğitilmiş makine öğrenimi modeli (pickle dosyası).
* **veriseti.csv:** Eğitim için kullanılan yüz işaret noktası verilerini içeren CSV dosyası.
* **face\_landmarker\_v2\_with\_blendshapes.task:** MediaPipe'in yüz işaret noktası tespiti için kullandığı model dosyası.

## Gereksinimler

Projenin çalışması için aşağıdaki kütüphanelerin ve araçların yüklü olması gerekmektedir:

* Python 3.x
* MediaPipe
* OpenCV (cv2)
* scikit-learn
* pandas
* numpy
* matplotlib

Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:

```bash
pip install mediapipe opencv-python scikit-learn pandas numpy matplotlib
