#--------------------------------------
#
#
# TODO: Poprawić komentarze
#
# Status: ?%
#--------------------------------------

import cv2
import numpy as np

def filter_blur(image, kernel_size=(5, 5)):
    """
    Stosuje filtr Gaussowski do wygładzenia obrazu.

    :param image: Obraz w kolorze lub skali szarości (np. wynik konwersji do grayscale)
    :param kernel_size: Rozmiar jądra filtra Gaussa (musi być nieparzysty, np. (5,5))
    :return: Wygładzony obraz
    """

    gaussian = cv2.GaussianBlur(image, kernel_size, 0)

    bilateral_n_gaussian = cv2.bilateralFilter(gaussian, 9, 75, 75)


    return bilateral_n_gaussian

def binarize_image(image):
    """Wczytuje obraz, konwertuje do skali szarości i binaryzuje"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Konwersja do skali szarości
    
    # Binaryzacja - metoda Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

def transform_morphologically(binary_image, kernel_size=(5, 5)):
    """
    Stosuje operacje morfologiczne: najpierw otwarcie (usunięcie szumu), potem zamknięcie (domknięcie krawędzi).

    :param binary_image: Obraz binarny (czarno-biały, np. po binaryzacji)
    :param kernel_size: Rozmiar jądra do operacji morfologicznych
    :return: Przetworzony obraz binarny
    """
    kernel = np.ones(kernel_size, np.uint8)

    # Otwarcie: usuwa drobne zakłócenia (erozja → dylacja)
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Zamknięcie: wypełnia przerwy w konturach (dylacja → erozja)
    closed_n_opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed_n_opened


def find_receipt_contour(binary_image):
    """Znajduje największy kontur (najpewniej paragon)"""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None  # Jeśli nie znaleziono konturów
    
    largest_contour = max(contours, key=cv2.contourArea)  # Wybieramy największy kontur
    
    return largest_contour

def crop_receipt(image):
    """Przygotowuje obraz: binaryzacja, znajdowanie konturu i wycinanie paragonu"""
    binary = binarize_image(image)
    contour = find_receipt_contour(binary)

    if contour is None:
        print("Nie znaleziono konturu paragonu.")
        return None

    # Tworzymy prostokąt na podstawie konturu
    x, y, w, h = cv2.boundingRect(contour)
    cropped_receipt = image[y:y+h, x:x+w]  # Wycinamy obszar paragonu

    return cropped_receipt

