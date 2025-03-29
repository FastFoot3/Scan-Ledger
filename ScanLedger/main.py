import cv2
from core.crop_receipt_on_image import binarize_image, filter_blur, transform_morphologically  # Import funkcji

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
    binarized_image = binarize_image(filtered_image) # Binaryzacja

    # Testowanie różnych wartości blockSize i C
    for block_size in [11, 21, 31, 41]:
        for C in [2, 5, 10, 15]:
            binarized_image = binarize_image(filtered_image, block_size, C) # Binaryzacja

            cv2.imwrite(f"bin{i}_size={block_size}_C={C}.png", binarized_image)


    #cv2.imshow(f"Binarized - {path}", binarized_image)  # Pokaż wynik
    #cv2.imwrite(f"bin{i}.png", binarized_image) # Zapisz wynik binaryzacji


#cv2.waitKey(0)  # Czekaj na klawisz
#cv2.destroyAllWindows()  # Zamknij okna po wyświetleniu
