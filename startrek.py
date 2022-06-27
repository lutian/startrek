import cv2
import mediapipe as mp
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
        self.char_fyou    = cv2.imread('assets/fyou.png', -1)

        # Se precisar mudo o tamanho da imagens
        # self.bg_image     = cv2.resize(self.bg_image, (self.w, self.h))
        self.char_spock   = cv2.resize(self.char_spock, (195, 450))
        self.char_kirk    = cv2.resize(self.char_kirk, (251, 560))
        self.char_khan    = cv2.resize(self.char_khan, (196, 470))
        self.char_fyou    = cv2.resize(self.char_fyou, (350, 350))

        # Defino a posição das imagens
        self.pos_bg       = [[0, self.h], [0, self.w]]
        # Defino as posições de cada caracter:
        # Para o plano Y defino a altura da tela - altura do caracter
        # Para o plano X defino uma posição + largura do caracter
        self.pos_spock = [[(self.h - 110 - self.char_spock.shape[0]), self.h - 110], [690, (690 + self.char_spock.shape[1])]]
        self.pos_kirk  = [[(self.h - 45 - self.char_kirk.shape[0]), self.h - 45], [430, (430 + self.char_kirk.shape[1])]]
        self.pos_khan  = [[(self.h - 95 - self.char_khan.shape[0]), self.h - 95], [290, (290 + self.char_khan.shape[1])]]

        self.pos_fyou  = [[((self.h / 2) - (self.char_fyou.shape[0] / 2)), ((self.h / 2) + (self.char_fyou.shape[0] / 2))], 
                        [((self.w / 2) - (self.char_fyou.shape[1] / 2)), ((self.w / 2) + (self.char_fyou.shape[1] / 2))]]

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

        self.WRIST = 0
        self.THUMB_IP = 3
        self.THUMB_TIP = 4
        self.INDEX_FINGER_MCP = 5
        self.INDEX_FINGER_PIP = 6
        self.INDEX_FINGER_DIP = 7
        self.INDEX_FINGER_TIP = 8
        self.MIDDLE_FINGER_MCP = 9
        self.MIDDLE_FINGER_PIP = 10
        self.MIDDLE_FINGER_DIP = 11
        self.MIDDLE_FINGER_TIP = 12
        self.RING_FINGER_MCP = 13
        self.RING_FINGER_PIP = 14
        self.RING_FINGER_DIP = 15
        self.RING_FINGER_TIP = 16
        self.PINKY_MCP = 17
        self.PINKY_PIP = 18
        self.PINKY_DIP = 19
        self.PINKY_TIP = 20

        self.update()

    def update(self):
        while True:
            with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as mp_holistic:
                # Carrego o fundo
                self.addBackground()

                if self.capture_front.isOpened():
                    # Read frame
                    (self.status_front, self.frame_front) = self.capture_front.read()
                    
                    if (self.status_front == False):
                        break

                    # Atualizo as dimensões do vídeo de acordo com a imagem de fundo
                    ycenter = int(self.h / 2)
                    y1 = ycenter - int(self.frame_front.shape[0] / 2) 
                    y2 = ycenter + int(self.frame_front.shape[0] / 2)
                    xcenter = int(self.w / 2)
                    x1 = xcenter - int(self.frame_front.shape[1] / 2) 
                    x2 = xcenter + int(self.frame_front.shape[1] / 2)
                    self.frame_front = cv2.flip(self.frame_front,1)
                    self.frame_webcam[y1:y2,x1:x2] = self.frame_front

                    # Transformo do BGR para RGB
                    self.frame_front_rgb = cv2.cvtColor(self.frame_webcam, cv2.COLOR_BGR2RGB)
                    results = mp_holistic.process(self.frame_front_rgb)

                    # Obtenho a imagem binaria
                    _, frame_front_th = cv2.threshold(results.segmentation_mask, 0.75, 255, cv2.THRESH_BINARY)

                    # Converto o data type
                    frame_front_th = frame_front_th.astype(np.uint8)

                    # Aplico o blur
                    # frame_front_th = cv2.medianBlur(frame_front_th, 13)
                    # Invirto a mascara
                    frame_front_th_invert = cv2.bitwise_not(frame_front_th)

                    # TODO: validar se a pessoa dá um like, nesse caso aparecem os famosos e tira uma foto
                    if results.right_hand_landmarks:
                        print(
                            f'Right Hand coordinates: ('
                            f'{results.right_hand_landmarks.landmark}, '
                        )
                        if (self.isFYouSignal(results.right_hand_landmarks.landmark)):
                            # Usuário está fazendo f*ckyou
                            self.avoidTakePicture = True
                            self.qrcode = None
                            self.addParticles = False

                        if (self.isStarTrekSignal(results.right_hand_landmarks.landmark)):
                            # usuário fez a saudação de star trek (_\\//)
                            self.addParticles = True
                            self.qrcode = None

                        if (self.isLike(results.right_hand_landmarks.landmark) and self.addParticles):
                            self.frame_back = None
                            self.qrcode = None
                            self.takePhoto = True
                            self.selfieSaved = False
                            self.TIMER = int(3)

                    if self.avoidTakePicture == True:
                        self.avoidTakeSelfie()
                    else:
                        # Se chamei os characteres para a ponte, carrego o vídeo de fundo com a transportação
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

                    # Obtenho a cena do fundo
                    frame_bg = cv2.bitwise_and(self.frame_back, self.frame_back, mask=frame_front_th_invert)

                    # Obtenho a cena da frente
                    frame_fg = cv2.bitwise_and(self.frame_webcam, self.frame_webcam, mask=frame_front_th)

                    # junto o Background com o Foreground
                    self.frame_final = cv2.add(frame_bg, frame_fg)

                    cv2.imshow('frame_final', self.frame_final)

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

    def isLike(self, points):
        """
         Args:
            points: landmarks from mediapipe

        Returns:
            boolean check if hand is in Like mode
         """
        return points[self.THUMB_TIP].x > points[self.THUMB_IP].x \
            and points[self.INDEX_FINGER_TIP].y > points[self.INDEX_FINGER_DIP].y \
            and points[self.MIDDLE_FINGER_TIP].y > points[self.MIDDLE_FINGER_DIP].y \
            and points[self.RING_FINGER_TIP].y > points[self.RING_FINGER_DIP].y \
             and points[self.PINKY_TIP].y > points[self.PINKY_DIP].y
            
    def isFistClosed(self, points):
        """
        Args:
            points: landmarks from mediapipe

        Returns:
            boolean check if fist is closed
        """
        return points[self.MIDDLE_FINGER_MCP].y < points[self.MIDDLE_FINGER_TIP].y \
            and points[self.PINKY_MCP].y < points[self.PINKY_TIP].y \
            and points[self.RING_FINGER_MCP].y < points[self.RING_FINGER_TIP].y


    def isHandDown(self, points):
        """
        Args:
            points: landmarks from mediapipe

        Returns:
            boolean check if hand is down i.e. inverted
        """
        return points[self.MIDDLE_FINGER_TIP].y > points[self.WRIST].y


    def isHandUp(self, points):
        """
        Args:
            points: landmarks from mediapipe

        Returns:
            boolean check if hand is up
        """
        return points[self.MIDDLE_FINGER_TIP].y < points[self.WRIST].y


    def isTwoSignal(self, points):
        """
        Args:
            points: landmarks from mediapipe

        Returns:
            boolean check if fingers show two
        """
        return points[self.INDEX_FINGER_TIP].y < points[self.INDEX_FINGER_PIP].y \
            and points[self.MIDDLE_FINGER_TIP].y < points[self.MIDDLE_FINGER_PIP].y \
            and points[self.RING_FINGER_PIP].y < points[self.RING_FINGER_TIP].y \
            and points[self.PINKY_PIP].y < points[self.PINKY_TIP].y \
            and points[self.INDEX_FINGER_PIP].x < points[self.INDEX_FINGER_MCP].x \
            and points[self.MIDDLE_FINGER_PIP].x > points[self.MIDDLE_FINGER_MCP].x \


    def isThreeSignal(self, points):
        """
        Args:
            points: landmarks from mediapipe

        Returns:
            boolean check if fingers show three
        """
        return points[self.INDEX_FINGER_TIP].y < points[self.INDEX_FINGER_PIP].y \
            and points[self.MIDDLE_FINGER_TIP].y < points[self.MIDDLE_FINGER_PIP].y \
            and points[self.RING_FINGER_PIP].y > points[self.RING_FINGER_TIP].y \
            and points[self.PINKY_PIP].y < points[self.PINKY_TIP].y


    def isFYouSignal(self, points):
        """
        Args:
            points: landmarks from mediapipe

        Returns:
            boolean check if is showing the middle finger
        """
        return points[self.INDEX_FINGER_PIP].y < points[self.INDEX_FINGER_TIP].y \
            and points[self.MIDDLE_FINGER_PIP].y > points[self.MIDDLE_FINGER_TIP].y \
            and points[self.RING_FINGER_PIP].y < points[self.RING_FINGER_TIP].y \
            and points[self.PINKY_PIP].y < points[self.PINKY_TIP].y


    def isStarTrekSignal(self, points):
        """
        Args:
            points: landmarks from mediapipe

        Returns:
            boolean check if fingers show four as star trek signal
        """
        return points[self.INDEX_FINGER_TIP].y < points[self.INDEX_FINGER_PIP].y \
            and points[self.MIDDLE_FINGER_TIP].y < points[self.MIDDLE_FINGER_PIP].y \
            and points[self.RING_FINGER_TIP].y < points[self.RING_FINGER_PIP].y \
            and points[self.PINKY_TIP].y < points[self.PINKY_PIP].y \
            and points[self.MIDDLE_FINGER_PIP].x < points[self.MIDDLE_FINGER_MCP].x \
            and points[self.RING_FINGER_PIP].x > points[self.RING_FINGER_MCP].x \


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

    def avoidTakeSelfie(self):
        self.addImageTransparent(self.pos_fyou, self.char_fyou, True)

if __name__ == '__main__':
    selfie_with_startrek = startrek()
