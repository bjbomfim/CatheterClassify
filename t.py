# import cv2
# import os
# import numpy as np

# img = cv2.imread("Resultado da pesquisa (1).jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (384, 384))

# _, binary_image = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

# # Fechamento morfológico para suavizar as linhas das mascaras
# kernel = np.ones((7,7), np.uint8)
# tube_complited = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# # Capturando o contorno dos tubos
# contours, _ = cv2.findContours(tube_complited, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# if contours:
#     contour = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(contour)
#     length = max(w, h)
#     width = min(w, h)
    
#     print(length)
#     print(width)
# else:
#     print("sem contorno")

# cv2.imshow('Imagem binarizada original', binary_image)
# cv2.imshow('Imagem após fechamento morfológico', tube_complited)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np
import cv2

# # Criar uma imagem preta em escala de cinza
# image = np.zeros((1080, 1080, 1), dtype=np.uint8)

# # Salvar a imagem como 'semtubo_gray.jpg'
# cv2.imwrite('semtubo.jpg', image)

# print("Imagem 'semtubo.jpg' criada com sucesso.")

imagem_raiox = cv2.imread("Resultado pesquisa.jpg")
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#refined_mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
imagem_raiox = cv2.resize(imagem_raiox, (512, 512))

mascara = cv2.imread('1.2.826.0.1.3680043.8.498.81392493433269832920698532113815126933.jpg', cv2.IMREAD_GRAYSCALE)

mascara_rgb = cv2.cvtColor(mascara, cv2.COLOR_GRAY2RGB)

# Aplicar a máscara na imagem de raio-X
imagem_segmentada = cv2.addWeighted(imagem_raiox, 1, mascara_rgb, 0.5, 0)

# Exibir a imagem final
cv2.imshow('Imagem Final', imagem_segmentada)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite("novo.jpg", imagem_final)