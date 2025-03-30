import cv2
from core.crop_receipt_on_image import binarize_image, filter_blur, transform_morphologically, detect_edges  # Import funkcji

# Lista ścieżek do obrazów
image_paths = [
    "ScanLedger/assets/sample_receipts/aldi.jpg",
    "ScanLedger/assets/sample_receipts/biedronka.jpg",
    "ScanLedger/assets/sample_receipts/zabka.jpg"
]

i = 0

for path in image_paths:
    i += 1
    image = cv2.imread(path)  # Wczytaj pojedynczy obraz

    if image is None:
        print(f"Błąd: Nie można wczytać obrazu {path}")
        continue  # Przejdź do następnego obrazu

    filtered_image = filter_blur(image)  # Przetwórz obraz

    # edges_image = detect_edges(filtered_image)  # Wykrywanie krawędzi

    # cv2.imwrite(f"edges{i}.png", edges_image) # Zapisz wynik binaryzacji

    # binarized_image = binarize_image(filtered_image) # Binaryzacja

    # Testowanie różnych wartości blockSize i C
    for low_thres in [75, 100, 150, 50, 67, 100]:
        for high_thres in [150, 200, 300, 150, 200, 300]:
            binarized_image = detect_edges(filtered_image, low_thres, high_thres) # Binaryzacja

            cv2.imwrite(f"bin{i}_low={low_thres}_high={high_thres}.png", binarized_image)


    #cv2.imshow(f"Binarized - {path}", binarized_image)  # Pokaż wynik
    #cv2.imwrite(f"bin{i}.png", binarized_image) # Zapisz wynik binaryzacji


#cv2.waitKey(0)  # Czekaj na klawisz
#cv2.destroyAllWindows()  # Zamknij okna po wyświetleniu
