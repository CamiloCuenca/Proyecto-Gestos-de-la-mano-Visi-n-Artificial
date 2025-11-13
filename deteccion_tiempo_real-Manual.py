import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Convertir ROI a espacio HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Rango de color de piel (ajustable según luz)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Crear máscara del color de piel
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Suavizar para eliminar ruido
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Buscar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        hull = cv2.convexHull(cnt)
        cv2.drawContours(roi, [cnt], -1, (255, 0, 0), 2)
        cv2.drawContours(roi, [hull], -1, (0, 255, 0), 2)

        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(cnt, hull_indices)
            if defects is not None:
                count_defects = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    a = np.linalg.norm(np.array(end) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(end) - np.array(far))
                    angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

                    if angle <= np.pi / 2:
                        count_defects += 1
                        cv2.circle(roi, far, 5, (0, 0, 255), -1)

                fingers = count_defects + 1
                if fingers == 1:
                    text = "Mano cerrada"
                elif fingers > 1:
                    text = f"{fingers} dedos levantados"
                else:
                    text = "No detectado"

                cv2.putText(frame, text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow("Detección de Mano (Color piel)", frame)
    cv2.imshow("Máscara de piel", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
