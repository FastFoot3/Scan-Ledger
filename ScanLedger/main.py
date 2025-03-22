import cv2
from core.crop_receipt_on_image import binarize_image  # Import funkcji binarizacji

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

    binarized_image = binarize_image(image)  # Przetwórz obraz

    #$cv2.imshow(f"Binarized - {path}", binarized_image)  # Pokaż wynik
    cv2.imwrite(f"bin{i}.png", binarized_image)


#cv2.waitKey(0)  # Czekaj na klawisz
#cv2.destroyAllWindows()  # Zamknij okna po wyświetleniu
