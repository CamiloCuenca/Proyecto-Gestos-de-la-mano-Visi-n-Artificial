import cv2
from cvzone.HandTrackingModule import HandDetector

# Inicializar cámara
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)  # Detección y dibujo

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        total = fingers.count(1)

        # Clasificación según cantidad de dedos levantados
        if total == 0:
            estado = " Mano cerrada"
        elif total == 5:
            estado = " Mano abierta"
        elif total == 1:
            estado = " 1 dedo levantado"
        elif total == 2:
            estado = " 2 dedos levantados"
        elif total == 3:
            estado = " 3 dedos levantados"
        elif total == 4:
            estado = " 4 dedos levantados"
        else:
            estado = "Gesto no reconocido"

        # Mostrar información en pantalla
        cv2.putText(img, f"Dedos levantados: {total}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, estado, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Detección de Mano - CVZone", img)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
