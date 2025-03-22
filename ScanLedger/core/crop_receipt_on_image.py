import cv2
import numpy as np


def binarize_image(image):
    """Wczytuje obraz, konwertuje do skali szarości i binaryzuje"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Konwersja do skali szarości
    
    # Binaryzacja - metoda Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


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

