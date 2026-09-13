# Laboratoria z uczenia maszynowego

Repozytorium zawiera komplet rozwiązań laboratoriów 1-7: klasyfikację, regresję, analizę skupień, sieć CNN w PyTorch oraz eksperymenty z MLflow i DVC. Każde laboratorium ma osobny skrypt uruchomieniowy i zapisuje wyniki w katalogu `labs/labXX/outputs/`.

## Struktura projektu

| Laboratorium | Temat | Skrypt | Dane wejściowe |
| --- | --- | --- | --- |
| Lab 1 | Benchmark klasyfikacji | `labs/lab01/train.py` | `data/Online Retail 2.xlsx` |
| Lab 2 | Optymalizacja klasyfikacji | `labs/lab02/optimize.py` | `data/Online Retail 2.xlsx` |
| Lab 3 | Benchmark regresji | `labs/lab03/regression.py` | `data/Online Retail 2.xlsx` |
| Lab 4 | Optymalizacja regresji | `labs/lab04/optimize_regression.py` | `data/Online Retail 2.xlsx` |
| Lab 5 | Analiza skupień klientów | `labs/lab05/clustering.py` | `data/Online Retail 2.xlsx` |
| Lab 6 | Klasyfikacja obrazów CNN | `labs/lab06/cnn.py` | `data/pcam_images` i pliki etykiet |
| Lab 7 | MLflow i DVC | `labs/lab07/train_mlflow.py` | `data/Online Retail 2.xlsx` |

## Instalacja i dane

Wymagany jest Python 3.10 lub nowszy. Zależności instaluje się z katalogu głównego repozytorium:

```powershell
python -m pip install -r requirements.txt
```

Wszystkie dane wejściowe znajdują się w katalogu `data/`:

- `data/Online Retail 2.xlsx` - arkusz używany w laboratoriach 1-5 i 7,
- `data/pcam_images/` - obrazy PatchCamelyon dla Lab 6,
- `data/train_labels.csv`, `data/validation_labels.csv` i `data/test_labels.csv` - podziały danych dla Lab 6.

Skrypty korzystają z tych ścieżek domyślnie, dlatego poniższe komendy można uruchamiać bez parametrów wskazujących dane. Parametry `--input` i `--data-dir` pozwalają opcjonalnie wybrać inne położenie.

## Lab 1 - benchmark klasyfikacji

Na podstawie aktywności klienta do 31 maja 2011 r. model przewiduje, czy klient dokona kolejnego zakupu w ciągu następnych 90 dni. Dane są czyszczone, agregowane do poziomu klienta, uzupełniane i kodowane. Wartości odstające cech numerycznych są przycinane według granic IQR wyznaczonych na zbiorze treningowym. Modelem bazowym jest regresja logistyczna.

```powershell
python labs/lab01/train.py
```

Wynik na zbiorze testowym: accuracy 0,654, balanced accuracy 0,653, F1 0,616 i ROC AUC 0,732.

## Lab 2 - optymalizacja klasyfikacji

Regresja logistyczna, las losowy, gradient boosting i k-NN są porównywane za pomocą pięciokrotnej walidacji krzyżowej. Najlepsza rodzina jest dostrajana przez grid search bez wykorzystywania zbioru testowego.

```powershell
python labs/lab02/optimize.py
```

Najlepsza była regresja logistyczna z `C=0.01` i solverem `liblinear`. Wynik testowy: accuracy 0,669, balanced accuracy 0,668, F1 0,640 i ROC AUC 0,730.

Tabela porównawcza zawiera średnią i odchylenie standardowe accuracy, balanced accuracy, F1 oraz ROC AUC. Skrypt zapisuje też predykcje i macierz pomyłek `[[204, 72], [108, 160]]`, dzięki czemu ocena nie opiera się wyłącznie na jednym wskaźniku.

## Lab 3 - benchmark regresji

Aktywność klienta do 31 maja 2011 r. służy do przewidywania łącznych wydatków w kolejnych 90 dniach. Modelem bazowym jest regresja liniowa, a zbiór zawiera 2718 klientów.

Wartości odstające cech numerycznych są wykrywane regułą `1.5 x IQR`. Transformator `IQRClipper` nie usuwa klientów, tylko przycina wartości poniżej `Q1 - 1.5 x IQR` i powyżej `Q3 + 1.5 x IQR`. Granice są dopasowywane wyłącznie na zbiorze treningowym i bez zmian stosowane do zbioru testowego.

```powershell
python labs/lab03/regression.py
```

Wynik benchmarku: MAE 749,24, RMSE 2566,47 i R2 -0,026. Ujemne R2 wskazuje, że prosty model liniowy nie radzi sobie z silnie skośnymi i nieregularnymi przyszłymi wydatkami lepiej niż predykcja średniej. Raport `outlier_report.csv` wykazuje przycięcie 1014 wartości cech w zbiorze treningowym i 287 w testowym. Skrypt zapisuje zarówno wykres reszt, jak i czytelniejszy wykres wartości rzeczywistych względem przewidywanych; ponieważ model ma wiele cech, nie istnieje jedna dwuwymiarowa „linia regresji” do naniesienia na dane.

## Lab 4 - optymalizacja regresji

Porównywane są regresja Ridge, las losowy, histogram gradient boosting i k-NN. Zmienna docelowa jest transformowana przez `log1p`, a najlepszy model jest wybierany i dostrajany na podstawie walidacji krzyżowej.

```powershell
python labs/lab04/optimize_regression.py
```

Najlepszy wynik uzyskała regresja Ridge z `alpha=100`: MAE 578,48, RMSE 2465,51 i R2 0,054. Względem Lab 3 oznacza to spadek MAE o 22,8% i RMSE o 3,9%, ale nadal tylko niewielką wyjaśnioną część zmienności. Raport reszt pokazuje silną prawostronną skośność (9,94) oraz korelację reszt z predykcją 0,419, więc założenia prostego, jednorodnego błędu nie są dobrze spełnione.

## Lab 5 - analiza skupień

Klienci są opisywani przez recency, częstotliwość i wartość zakupów oraz liczbę produktów. K-means, Gaussian mixture i grupowanie aglomeracyjne są porównywane dla 2-8 klastrów z użyciem metryk silhouette, Calinskiego-Harabasza i Daviesa-Bouldina.

```powershell
python labs/lab05/clustering.py
```

Dla 4338 klientów wybrano dwa klastry i algorytm K-means z wynikiem silhouette 0,359. Pierwsze dwie składowe PCA wyjaśniają 83,1% wariancji przekształconych danych. Wynik silhouette oznacza umiarkowany, a nie bardzo wyraźny podział; profile klastrów należy zatem traktować jako segmentację opisową.

## Lab 6 - CNN w PyTorch

Sieć konwolucyjna klasyfikuje obrazy z dostarczonego podzbioru PatchCamelyon. Potok obejmuje zmianę rozmiaru do 96 x 96, augmentację zbioru treningowego, normalizację i automatyczny wybór CPU lub CUDA. Architektura `SimpleCNN` ma trzy warstwy konwolucyjne (32, 64 i 128 filtrów), trzy warstwy max-pooling oraz klasyfikator 18432 -> 256 -> 1. Batch size wynosi 8 dla treningu i 32 dla walidacji oraz testu, zgodnie z instrukcją.

```powershell
python labs/lab06/cnn.py
```

Po pięciu epokach wynik testowy wynosi: accuracy 0,795, precision 0,888, recall 0,689, F1 0,776 i ROC AUC 0,905. Macierz pomyłek to `[[88, 9], [32, 71]]`. Accuracy przewyższa bazową klasę większościową (0,515), ale niższy recall pokazuje, że model nadal pomija część przypadków pozytywnych. Najniższa strata walidacyjna wystąpiła w epoce 4; jej wzrost w epoce 5 jest lekkim sygnałem przeuczenia.

## Lab 7 - MLflow i DVC

Klasyfikator z wcześniejszych laboratoriów jest uruchamiany dla trzech wartości parametru `C`. Parametry, metryki i artefakty są rejestrowane w lokalnym eksperymencie MLflow, a dane i etap treningu są śledzone przez DVC.

Eksperyment MLflow można uruchomić bezpośrednio z katalogu głównego:

```powershell
python labs/lab07/train_mlflow.py
mlflow ui --backend-store-uri sqlite:///labs/lab07/mlflow.db
```

Etap DVC korzysta ze śledzonej kopii arkusza w `labs/lab07/data/`:

```powershell
Set-Location labs\lab07
dvc repro
```

Interfejs MLflow jest dostępny domyślnie pod adresem `http://127.0.0.1:5000`. Zapisane są dokładnie trzy biegi: `C=0.1`, `C=1.0` i `C=10.0`. Najwyższy ROC AUC, równy 0,732, uzyskano dla `C=1.0`; najwyższy F1 (0,633) uzyskało `C=0.1`, co pokazuje zależność wyboru od głównej metryki.

## Powtarzalność wyników

W eksperymentach stosowany jest `random_state=42`. W zadaniach nadzorowanych przetwarzanie jest dopasowywane wyłącznie na zbiorze treningowym, wybór modelu odbywa się przez walidację krzyżową, a zbiór testowy służy tylko do końcowej oceny.
