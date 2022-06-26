import cv2
import numpy as np
import os
import ftplib
import random
import string
import qrcode
import time

class startrek(object):

    def __init__(self):

        self.w = 1536
        self.h = 830

        # Importo as imagens
        self.bg_image     = cv2.imread('assets/bg_transporter.jpg', -1)
        self.char_spock   = cv2.imread('assets/spock.png', -1)
        self.char_kirk    = cv2.imread('assets/kirk.png', -1)
        self.char_khan    = cv2.imread('assets/khan.png', -1)

        # Se precisar mudo o tamanho da imagens
        # self.bg_image     = cv2.resize(self.bg_image, (self.w, self.h))
        self.char_spock   = cv2.resize(self.char_spock, (195, 450))
        self.char_kirk    = cv2.resize(self.char_kirk, (251, 560))
        self.char_khan    = cv2.resize(self.char_khan, (196, 470))

        # Defino a posição das imagens
        self.pos_bg       = [[0, self.h], [0, self.w]]
        # Defino as posições de cada caracter:
        # Para o plano Y defino a altura da tela - altura do caracter
        # Para o plano X defino uma posição + largura do caracter
        self.pos_spock = [[(self.h - 110 - self.char_spock.shape[0]), self.h - 110], [690, (690 + self.char_spock.shape[1])]]
        self.pos_kirk    = [[(self.h - 45 - self.char_kirk.shape[0]), self.h - 45], [430, (430 + self.char_kirk.shape[1])]]
        self.pos_khan    = [[(self.h - 95 - self.char_khan.shape[0]), self.h - 95], [290, (290 + self.char_khan.shape[1])]]

        self.qrcode = None
        self.pos_qrcode = [[self.h - 320, self.h - 20], [self.w - 320, self.w - 20]]

        self.frame_webcam = np.ones((self.h,self.w,3),dtype='uint8')*0
        self.frame_final = None
        self.image_path = 'temp'
        self.image_name = None
        self.image_extension = '.jpg'
        self.takePhoto = False

        # FTP account
        self.url = 'https://your.domain/?im='
        self.ftp_server = 'ftp.your.domain'
        self.ftp_user = 'user@your.domain'
        self.ftp_pass = 'password'

        self.TIMER = int(3)
        self.prev = None
        self.cur = None
        self.frame_timer = None
        self.selfieSaved = False

        self.cap_front = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.frame_front = None

        self.cap_particles = cv2.VideoCapture("assets/startrek.mp4")
        self.addParticles = False
        self.transported = False

        self.update()

    def update(self):
        while True:
            # Carrego o fundo
            self.addBackground()

            # Se chamei os characteres para a ponte, carrego o vídeo de fundo com com a transportação
            if self.addParticles == True:
                ret_particles, self.frame_particles = self.cap_particles.read()
                if ret_particles == True:
                    # Carrego o vídeo com a transportação com green screen
                    self.mergeParticles()
                else:
                    self.transported = True

            # carrego os caracteres
            if self.transported == True:
                self.addImages()

                # Carrego minha imagem desde a webcam
                #-#ret, self.frame_front = self.cap_front.read()
                #-#if ret == True:
                #-#    self.getContoursWebCam()

            # Carrego o timer para dar 3 segundos para se ajeitar para a selfie
            if (self.takePhoto):
                self.addTimer()
                if (self.TIMER == 0):
                    # Capturo a tela
                    self.takeScreenShot()

                if (self.selfieSaved):
                    # Salvo a foto
                    self.downloadImages()
                    # Envio via FTP para o server
                    self.ftpUploadSelfie()
                    # Gero o QR Code
                    self.genQrCode()
                    self.transported = False

            # Aplico filtros a cena (opcional)
            self.addFilters()

            # Apresento a cena
            cv2.imshow('frame_back', self.frame_back)

            key = cv2.waitKey(1)
            if key == ord('z'):
                # Chamo os characteres para a ponte de transportação
                self.qrcode = None
                self.addParticles = True
            if key == ord('f'):
                # Tiro a foto (começa o tomer de 3 segundos)
                if self.addParticles == True:
                    self.frame_back = None
                    self.qrcode = None
                    self.takePhoto = True
                    self.selfieSaved = False
                    self.TIMER = int(3)
            if key == ord('q'):
                # Fecho o script
                cv2.destroyAllWindows()
                exit(1)

    def addTimer(self):
        # teempo inicial
        if(self.prev == None): self.prev = time.time()

        self.frame_timer = self.frame_back.copy()

        # Coloco um fundo circular para ver melhor o timer (Opcional)
        #cv2.circle(self.frame_timer, (120, 125), 106, (90, 0, 0), -1) 
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(self.frame_timer, str(self.TIMER),
                                (50, 200), font,
                                7, (208, 185, 35),
                                10) #, cv2.LINE_AA
        cv2.imshow('frame_back', self.frame_timer) #
        cv2.waitKey(1)

        # tempo atual
        self.cur = time.time()

        # Resto o tempo atual com o inicial
        # se for maior ou igual a 1 resto 1 ao timer
        if self.cur-self.prev >= 1:
            self.prev = self.cur
            self.TIMER -= 1

    def addBackground(self):
        self.frame_back = np.ones((self.h,self.w,3),dtype='uint8')*255
        self.addImage(self.pos_bg, self.bg_image)

    def addImages(self):
        self.addImageTransparent(self.pos_kirk, self.char_kirk, True)
        self.addImageTransparent(self.pos_spock, self.char_spock, True)
        self.addImageTransparent(self.pos_khan, self.char_khan, True)

        if (self.qrcode is not None):
            self.addImage(self.pos_qrcode, self.qrcode)

    def zoom(self, img, zoom_factor=2):
        return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

    def shakeChar(self, pos, maxY=0, maxX=1):
        diffY = random.randint(0, maxY)
        diffX = random.randint(0, maxX)
        return [[(pos[0][0] - diffY), (pos[0][1] - diffY)], [(pos[1][0] + diffX), (pos[1][1] + diffX)]]

    def addImage(self, pos, image):
        self.frame_back[pos[0][0]:pos[0][1], pos[1][0]:pos[1][1]] = image

    def addImageTransparent(self, pos, image, shake = False):
        if(shake): pos = self.shakeChar(pos, 0, 1)
        self.frame_back[pos[0][0]:pos[0][1], pos[1][0]:pos[1][1]] = self.blendTransparent(
            self.frame_back[pos[0][0]:pos[0][1], pos[1][0]:pos[1][1]], image)

    def blendTransparent(self, face_img, overlay_t_img):
        # Split out the transparency mask from the colour info
        overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
        overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

        # Again calculate the inverse mask
        background_mask = 255 - overlay_mask

        # Turn the masks into three channel, so we can use them as weights
        overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

        # Create a masked out face image, and masked out overlay
        # We convert the images to floating point in range 0.0 - 1.0
        face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
        overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

        # And finally just add them together, and rescale it back to an 8bit integer image
        return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

    def getImageName(self):
        lower = string.ascii_lowercase
        num = string.digits
        length = 16
        temp = random.sample(lower + num, length)
        self.image_name = 'startrek_' + "".join(temp)

    def takeScreenShot(self):
        self.frame_final = self.frame_back.copy()
        cv2.rectangle(self.frame_back, (1, 1), (self.frame_back.shape[1]-1, self.frame_back.shape[0]-1), (255, 255, 255), -1)
        self.getImageName()
        self.selfieSaved = True

    def downloadImages(self):
        # Check whether the specified path exists or not
        isExist = os.path.exists(self.image_path)
        if not isExist:
            os.makedirs(self.image_path)
        cv2.imwrite(self.image_path + "/" + self.image_name + self.image_extension, self.frame_final)

    def ftpUploadSelfie(self):
        session = ftplib.FTP(str(self.ftp_server),str(self.ftp_user),str(self.ftp_pass))
        file = open(str(self.image_path + "/" + self.image_name + self.image_extension),'rb')                  # file to send
        session.storbinary('STOR ' + str(self.image_name + self.image_extension), file)     # send the file
        file.close()                                    # close file and FTP
        session.quit()
        os.remove(str(self.image_path + "/" + self.image_name + self.image_extension))

    def genQrCode(self):
        input_data = str(self.url + self.image_name)
        qr = qrcode.QRCode(
                version=1,
                box_size=10,
                border=5)
        qr.add_data(input_data)
        qr.make(fit=True)
        img = qr.make_image(fill='black', back_color='white')
        img_qr_path = self.image_path + '/qrcode.png'
        img.save(str(img_qr_path))
        img_qr = cv2.imread(str(img_qr_path), cv2.IMREAD_GRAYSCALE)
        img_qr = cv2.resize(img_qr, (300, 300))
        self.qrcode = cv2.cvtColor(img_qr, cv2.COLOR_GRAY2BGR)
        os.remove(str(self.image_path + '/qrcode.png'))
        self.takePhoto = False

    def addFilters(self):
        gaussianBlurKernel = np.array(([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), np.float32)/9
        sharpenKernel = np.array(([[0, -1, 0], [-1, 9, -1], [0, -1, 0]]), np.float32)/9
        meanBlurKernel = np.ones((3, 3), np.float32)/9

        gaussianBlur = cv2.filter2D(src=self.frame_back, kernel=gaussianBlurKernel, ddepth=-1)
        meanBlur = cv2.filter2D(src=self.frame_back, kernel=meanBlurKernel, ddepth=-1)
        sharpen = cv2.filter2D(src=self.frame_back, kernel=sharpenKernel, ddepth=-1)

        self.frame_back = np.concatenate((self.frame_back, gaussianBlur, meanBlur, sharpen), axis=1)

    def mergeParticles(self):
        self.frame_particles = cv2.resize(self.frame_particles, (self.w, self.h))

        u_green = np.array([120, 255, 120])
        l_green = np.array([0, 10, 0])

        mask = cv2.inRange(self.frame_particles, l_green, u_green)
        res = cv2.bitwise_and(self.frame_particles, self.frame_particles, mask = mask)

        f = self.frame_particles - res
        self.frame_back = np.where(f == 0, self.frame_back, f)

        #cv2.imshow("video", frame)
        #cv2.imshow("mask", f)

    def getContoursWebCam(self):
        # Atualizo as dimensões do vídeo de acordo com a imagem de fundo
        #ycenter = int(self.h / 2)
        #y1 = ycenter - int(self.frame_front.shape[0] / 2)
        #y2 = ycenter + int(self.frame_front.shape[0] / 2)
        #xcenter = int(self.w / 2)
        #x1 = xcenter - int(self.frame_front.shape[1] / 2)
        #x2 = xcenter + int(self.frame_front.shape[1] / 2)
        self.frame_front = cv2.flip(self.frame_front,1)
        #self.frame_webcam[y1:y2,x1:x2] = self.frame_front

        self.frame_webcam = cv2.resize(self.frame_front, (self.w, self.h))

        u_green = np.array([130, 160, 200])
        l_green = np.array([0, 0, 0])

        mask = cv2.inRange(self.frame_webcam, l_green, u_green)
        res = cv2.bitwise_and(self.frame_webcam, self.frame_webcam, mask = mask)

        f = self.frame_webcam - res
        self.frame_back = np.where(f == 0, self.frame_back, f)

if __name__ == '__main__':
    selfie_with_startrek = startrek()
