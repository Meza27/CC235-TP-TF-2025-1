from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.graphics import Color, RoundedRectangle, Rectangle
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup

import cv2
import numpy as np
import pickle
import os

# Importaciones de PyTorch con manejo de errores para Android
try:
    import torch
    from torchvision import transforms
    PYTORCH_AVAILABLE = True
    print("PyTorch importado correctamente")
except ImportError as e:
    print(f"PyTorch no disponible: {e}")
    PYTORCH_AVAILABLE = False
    torch = None
    transforms = None

# Configuración visual de la aplicación
Window.clearcolor = (0, 0, 0, 1)  # Fondo negro
IMG_SIZE = 224 

# Configuración de dispositivo 
if PYTORCH_AVAILABLE:
    device = torch.device("cpu")  # Forzar CPU para mayor compatibilidad en Android
else:
    device = None

# Variables globales para el modelo y clasificación
MODEL_LOADED = False
model = None
# Lista de personajes que el modelo puede detectar (Street Fighter)
class_names = ['Ryu', 'Chun-Li', 'Ken', 'Blanka', 'Zangief', 'Akuma', 'Balrog', 'Cammy', 'Dhalsim', 'E.Honda', 'Guile', 'M.Bison', 'Sagat', 'Sakura', 'Vega']

try:
    # Intentar cargar las clases desde archivo pickle
    if os.path.exists("class_names.pkl"):
        with open("class_names.pkl", "rb") as f:
            class_names = pickle.load(f)
        print(f"Clases cargadas: {len(class_names)} personajes")
    else:
        print("Archivo class_names.pkl no encontrado, usando clases por defecto")
    
    # Cargar el modelo preentrenado si está disponible
    if PYTORCH_AVAILABLE and os.path.exists("cnn_model_mobile.pt"):
        model = torch.jit.load("cnn_model_mobile.pt", map_location=device)
        model.eval() 
        MODEL_LOADED = True
        print("Modelo PyTorch cargado correctamente")
    else:
        print("Modelo no disponible o PyTorch no instalado")
        
except Exception as e:
    print(f"Error cargando recursos: {e}")
    MODEL_LOADED = False

# Transformaciones necesarias para preprocesar las imágenes antes de la predicción
if PYTORCH_AVAILABLE:
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convertir a PIL Image
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Redimensionar a 224x224
        transforms.ToTensor(),  # Convertir a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
else:
    transform = None

# Pantalla de carga
class SplashScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20)
        
        # Logo 
        try:
            logo = Image(source='logo.png', allow_stretch=True)
        except:
            logo = Label(text='STREET FIGHTER\nDETECTOR', font_size='24sp', 
                        halign='center', color=(1, 0.5, 0, 1))
        
        layout.add_widget(Widget())
        layout.add_widget(logo)
        layout.add_widget(Widget())
        self.add_widget(layout)

# Pantalla principal
class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)

        # Encabezado
        header = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        
        # Icono estrella 
        try:
            star_icon = Image(source='star_icon.png', size_hint=(None, None), size=(30, 30))
        except:
            star_icon = Label(text='★', font_size='20sp', size_hint=(None, None), size=(30, 30))
        header.add_widget(star_icon)
        
        # Título
        title_label = Label(text='Detector Street Fighter', font_size='20sp', bold=True, color=(1, 0.5, 0, 1))
        try:
            title_label.font_name = 'SSF4 ABUKET.ttf'
        except:
            pass 
        header.add_widget(title_label)
        
        # Icono de ayuda 
        try:
            help_icon = Image(source='question_icon.png', size_hint=(None, None), size=(30, 30))
        except:
            help_icon = Label(text='?', font_size='20sp', size_hint=(None, None), size=(30, 30))
        help_icon.bind(on_touch_down=self.on_help_icon_press)
        header.add_widget(help_icon)
        layout.add_widget(header)
        
        # Línea separadora del header
        separator = Widget(size_hint_y=None, height=2)
        with separator.canvas:
            Color(1, 1, 1, 1)
            self.separator_rect = Rectangle(pos=(0, 0), size=(Window.width, 2))
        
        def update_separator(instance, value):
            self.separator_rect.pos = (instance.x, instance.y)
            self.separator_rect.size = (instance.width, 2)
        
        separator.bind(pos=update_separator, size=update_separator)
        layout.add_widget(separator)

        # Botones centrados 
        btn_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.8), spacing=20)
        btn_layout.add_widget(Widget())  # Espaciador superior
        
        # === BOTÓN DE CÁMARA (DETECCIÓN EN TIEMPO REAL) ===
        btn_horizontal_camera = BoxLayout(orientation='horizontal', size_hint=(1, None), height=200)
        btn_horizontal_camera.add_widget(Widget())  # Centrar horizontalmente

        # Crear el botón de cámara como un contenedor personalizado
        btn_camera = BoxLayout(orientation='vertical', spacing=10, padding=20,
                             size_hint=(None, None), size=(300, 200))
        
        # Fondo rojo para el botón de cámara
        with btn_camera.canvas.before:
            Color(0.6, 0.1, 0.1, 1)  # Color rojo oscuro
            self.rect_camera = RoundedRectangle(size=btn_camera.size, pos=btn_camera.pos, radius=[10])
        
        btn_camera.bind(size=self.update_rect_camera, pos=self.update_rect_camera)
        
        # Icono de cámara 
        try:
            camera_icon = Image(source='camera_icon.png', size_hint=(None, None), size=(60, 60), 
                          pos_hint={'center_x': 0.5})
        except:
            camera_icon = Label(text='📷', font_size='40sp', size_hint=(None, None), size=(60, 60))
        btn_camera.add_widget(camera_icon)
        
        # Texto del botón de cámara
        camera_label = Label(text='Abrir Cámara\nLive character detection', font_size='14sp', 
                           halign='center', valign='center', color=(1,1,1,1))
        try:
            camera_label.font_name = 'SSF4 ABUKET.ttf'
        except:
            pass
        btn_camera.add_widget(camera_label)
        
        btn_camera.bind(on_touch_down=self.on_camera_button_press)

        btn_horizontal_camera.add_widget(btn_camera)
        btn_horizontal_camera.add_widget(Widget())
        
        btn_layout.add_widget(btn_horizontal_camera)

        # === BOTÓN DE SELECCIONAR IMAGEN (ANÁLISIS ESTÁTICO) ===
        btn_horizontal_image = BoxLayout(orientation='horizontal', size_hint=(1, None), height=200)
        btn_horizontal_image.add_widget(Widget())  # Centrar horizontalmente

        # Crear el botón de imagen como un contenedor personalizado
        btn_image = BoxLayout(orientation='vertical', spacing=10, padding=20,
                            size_hint=(None, None), size=(300, 200))
        
        # Fondo azul para el botón de imagen
        with btn_image.canvas.before:
            Color(0.1, 0.4, 0.6, 1)  # Color azul
            self.rect_image = RoundedRectangle(size=btn_image.size, pos=btn_image.pos, radius=[10])
        
        btn_image.bind(size=self.update_rect_image, pos=self.update_rect_image)
        
        # Icono de imagen 
        try:
            image_icon = Image(source='carpeta_icon.png', size_hint=(None, None), size=(60, 60), 
                          pos_hint={'center_x': 0.5})
        except:
            image_icon = Label(text='🖼️', font_size='40sp', size_hint=(None, None), size=(60, 60))
        btn_image.add_widget(image_icon)
        
        # Texto del botón de imagen
        image_label = Label(text='Seleccionar Imagen\nImage character detection', font_size='14sp', 
                          halign='center', valign='center', color=(1,1,1,1))
        try:
            image_label.font_name = 'SSF4 ABUKET.ttf'
        except:
            pass
        btn_image.add_widget(image_label)
        
        btn_image.bind(on_touch_down=self.on_image_button_press)

        btn_horizontal_image.add_widget(btn_image)
        btn_horizontal_image.add_widget(Widget())
        
        btn_layout.add_widget(btn_horizontal_image)
        btn_layout.add_widget(Widget())
        layout.add_widget(btn_layout)

        self.add_widget(layout)

    def update_rect_camera(self, instance, value):
        self.rect_camera.pos = instance.pos
        self.rect_camera.size = instance.size

    def update_rect_image(self, instance, value):
        self.rect_image.pos = instance.pos
        self.rect_image.size = instance.size

    def on_camera_button_press(self, instance, touch):
        if instance.collide_point(*touch.pos):
            self.abrir_camara(instance)
            return True
        return False

    def on_image_button_press(self, instance, touch):
        if instance.collide_point(*touch.pos):
            self.abrir_selector_imagen(instance)
            return True
        return False

    def on_help_icon_press(self, instance, touch):
        if instance.collide_point(*touch.pos):
            self.manager.current = 'ayuda'
            return True
        return False

    def abrir_camara(self, instance):
        self.manager.current = 'camara'

    def abrir_selector_imagen(self, instance):
        """Función que abre un popup con explorador de archivos para seleccionar imágenes"""
        # Crear el contenido del popup
        content = BoxLayout(orientation='vertical', spacing=10)
        
        # Selector de archivos con filtros para imágenes
        filechooser = FileChooserIconView(
            filters=['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif'],  # Solo archivos de imagen
            # Intentar abrir la carpeta de fotos del dispositivo Android
            path='/storage/emulated/0/Pictures' if os.path.exists('/storage/emulated/0/Pictures') else os.path.expanduser('~')
        )
        content.add_widget(filechooser)
        
        # Botones de acción
        buttons = BoxLayout(orientation='horizontal', size_hint_y=None, height=50, spacing=10)
        
        select_btn = Button(text='Seleccionar', size_hint_x=0.5)
        cancel_btn = Button(text='Cancelar', size_hint_x=0.5)
        
        buttons.add_widget(select_btn)
        buttons.add_widget(cancel_btn)
        content.add_widget(buttons)
        
        # Crear y mostrar popup
        popup = Popup(
            title='Seleccionar Imagen',
            content=content,
            size_hint=(0.9, 0.9) 
        )
        
        def seleccionar_imagen(instance):
            """Función que se ejecuta cuando el usuario selecciona una imagen"""
            if filechooser.selection:
                image_path = filechooser.selection[0]
                popup.dismiss()
                # Navegar a la pantalla de análisis de imagen
                image_screen = self.manager.get_screen('imagen')
                image_screen.cargar_imagen(image_path)
                self.manager.current = 'imagen'
        
        def cancelar(instance):
            """Cerrar el popup sin hacer nada"""
            popup.dismiss()
        
        # Conectar los botones con sus funciones
        select_btn.bind(on_press=seleccionar_imagen)
        cancel_btn.bind(on_press=cancelar)
        
        popup.open()

# Pantalla de cámara con detección real
class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        
        # Botón de volver
        top_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50, padding=10)
        back_btn = Button(text='Volver', size_hint=(None, None), size=(80, 40),
                         background_color=(0.6, 0.1, 0.1, 1), color=(1, 1, 1, 1))
        back_btn.bind(on_press=self.volver_inicio)
        top_layout.add_widget(back_btn)
        top_layout.add_widget(Widget())
        layout.add_widget(top_layout)
        
        # Imagen de la cámara
        self.img = Image()
        layout.add_widget(self.img)
        
        # Label de detección
        self.label = Label(text="Iniciando cámara...", size_hint=(1, 0.1), font_size='18sp')
        layout.add_widget(self.label)
        
        self.add_widget(layout)
        self.capture = None

    def on_enter(self):
        """Función que se ejecuta al entrar a la pantalla de cámara - inicializa la cámara"""
        try:
            # Abrir la cámara (índice 0 = cámara principal)
            self.capture = cv2.VideoCapture(2)
            
            # Configurar resolución de cámara para mejor calidad
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            # Verificar qué resolución obtuvimos realmente
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            
            # Si no se consigue 720p, intentar con 1080p 
            if actual_width < 1280 or actual_height < 720:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Optimizaciones adicionales para dispositivos móviles
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
            
            if self.capture.isOpened():
                # Iniciar el loop de actualización de la cámara a 30 FPS
                Clock.schedule_interval(self.update, 1.0 / 30.0)
                self.label.text = f"Cámara HD: {actual_width}x{actual_height} @ {actual_fps}fps"
            else:
                self.label.text = "Error: No se pudo abrir la cámara"
        except Exception as e:
            self.label.text = f"Error de cámara: {str(e)[:30]}"

    def update(self, dt):
        """Función principal que se ejecuta cada frame para procesar la imagen de la cámara"""
        try:
            # Verificar que la cámara esté disponible
            if self.capture is None or not self.capture.isOpened():
                self.label.text = "Cámara no disponible"
                return

            # Capturar un frame de la cámara
            ret, frame = self.capture.read()
            if not ret:
                self.label.text = "No se puede leer de la cámara"
                return

            # DETECCIÓN DE PERSONAJES 
            if MODEL_LOADED and model is not None and PYTORCH_AVAILABLE:
                try:
                    # Preprocesar la imagen para el modelo
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
                    input_tensor = transform(img_rgb).unsqueeze(0).to(device)  # Aplicar transformaciones

                    # Realizar la predicción con el modelo
                    with torch.no_grad():  
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)  # Convertir a probabilidades
                        pred_idx = torch.argmax(probs, dim=1).item()  # Índice de la clase más probable
                        confianza = probs[0, pred_idx].item()  # Nivel de confianza
                        
                        # Obtener el nombre del personaje detectado
                        if pred_idx < len(class_names):
                            personaje = class_names[pred_idx]
                        else:
                            personaje = "Desconocido"

                    # Mostrar el resultado
                    self.label.text = f"{personaje} ({confianza*100:.2f}%)"
                    
                except Exception as e:
                    # Si hay error en la detección
                    self.label.text = f"Error en detección: {str(e)[:20]}"
            else:
                # Modo simulación si el modelo no está disponible
                import random
                personaje = random.choice(class_names)
                confianza = random.uniform(60, 95)
                status = "Sin modelo" if not PYTORCH_AVAILABLE else "Modelo no cargado"
                self.label.text = f"{personaje} ({confianza:.1f}%) - {status}"

            # Mostrar el frame en la interfaz 
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture
            
        except Exception as e:
            self.label.text = f"Error: {str(e)[:30]}"

    def on_leave(self):
        """Limpiar recursos al salir de la pantalla"""
        Clock.unschedule(self.update)
        if hasattr(self, 'capture') and self.capture is not None:
            try:
                self.capture.release()
                self.capture = None
            except Exception as e:
                print(f"Error liberando cámara: {e}")

    def volver_inicio(self, instance):
        self.manager.current = 'inicio'

# Pantalla de análisis de imagen
class ImageScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        
        # Botón de volver
        top_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50, padding=10)
        back_btn = Button(text='Volver', size_hint=(None, None), size=(80, 40),
                         background_color=(0.6, 0.1, 0.1, 1), color=(1, 1, 1, 1))
        back_btn.bind(on_press=self.volver_inicio)
        top_layout.add_widget(back_btn)
        
        # Botón para seleccionar otra imagen
        another_btn = Button(text='Otra Imagen', size_hint=(None, None), size=(120, 40),
                           background_color=(0.1, 0.4, 0.6, 1), color=(1, 1, 1, 1))
        another_btn.bind(on_press=self.seleccionar_otra_imagen)
        top_layout.add_widget(another_btn)
        
        top_layout.add_widget(Widget())
        layout.add_widget(top_layout)
        
        # Imagen seleccionada
        self.img = Image()
        layout.add_widget(self.img)
        
        # Label de detección
        self.label = Label(text="Selecciona una imagen", size_hint=(1, 0.1), font_size='18sp')
        layout.add_widget(self.label)
        
        self.add_widget(layout)
        self.current_image_path = None

    def cargar_imagen(self, image_path):
        """Función principal para cargar una imagen seleccionada y realizar detección de personajes"""
        try:
            self.current_image_path = image_path
            
            # Cargar la imagen usando OpenCV para procesamiento
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                self.label.text = "Error: No se pudo cargar la imagen"
                return
            
            # Mostrar la imagen en la interfaz de usuario
            self.img.source = image_path
            
            # PROCESO DE DETECCIÓN DE PERSONAJES EN LA IMAGEN
            if MODEL_LOADED and model is not None and PYTORCH_AVAILABLE:
                try:
                    # Preprocesar la imagen para que sea compatible con el modelo
                    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # Convertir colores
                    input_tensor = transform(img_rgb).unsqueeze(0).to(device)  # Aplicar transformaciones

                    # Ejecutar el modelo de inteligencia artificial
                    with torch.no_grad():  
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)  # Obtener probabilidades
                        pred_idx = torch.argmax(probs, dim=1).item()  # Clase más probable
                        confianza = probs[0, pred_idx].item()  # Nivel de confianza
                        
                        # Traducir el índice a nombre de personaje
                        if pred_idx < len(class_names):
                            personaje = class_names[pred_idx]
                        else:
                            personaje = "Desconocido"

                    # Mostrar el resultado de la detección
                    self.label.text = f"Detectado: {personaje} ({confianza*100:.2f}%)"
                    
                except Exception as e:
                    self.label.text = f"Error en detección: {str(e)[:30]}"
            else:
                # Modo de demostración si no hay modelo disponible
                import random
                personaje = random.choice(class_names)
                confianza = random.uniform(60, 95)
                status = "Sin modelo" if not PYTORCH_AVAILABLE else "Modelo no cargado"
                self.label.text = f"Detectado: {personaje} ({confianza:.1f}%) - {status}"
                
        except Exception as e:
            self.label.text = f"Error cargando imagen: {str(e)[:30]}"

    def volver_inicio(self, instance):
        self.manager.current = 'inicio'

    def seleccionar_otra_imagen(self, instance):
        # Volver a la pantalla principal para seleccionar otra imagen
        home_screen = self.manager.get_screen('inicio')
        home_screen.abrir_selector_imagen(instance)

# Pantalla de ayuda
class HelpScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)

        # Título
        header = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        title_label = Label(text='Detector Street Fighter', font_size='20sp', bold=True, 
                          color=(1, 0.5, 0, 1), halign='center')
        try:
            title_label.font_name = 'SSF4 ABUKET.ttf'
        except:
            pass
        header.add_widget(title_label)
        layout.add_widget(header)

        # Separador
        separator = Widget(size_hint_y=None, height=2)
        with separator.canvas:
            Color(1, 1, 1, 1)
            self.separator_rect = Rectangle(pos=(0, 0), size=(Window.width, 2))
        
        def update_separator(instance, value):
            self.separator_rect.pos = (instance.x, instance.y)
            self.separator_rect.size = (instance.width, 2)
        
        separator.bind(pos=update_separator, size=update_separator)
        layout.add_widget(separator)

        # Logo centrado 
        logo_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=120)
        logo_layout.add_widget(Widget())
        try:
            logo = Image(source='logo.png', size_hint=(None, None), size=(120, 120))
        except:
            logo = Label(text='🥊', font_size='60sp', size_hint=(None, None), size=(120, 120))
        logo_layout.add_widget(logo)
        logo_layout.add_widget(Widget())
        layout.add_widget(logo_layout)

        # Información
        info_text = "Aplicación para detectar personajes de Street Fighter\nusando la cámara del dispositivo."
        info_label = Label(text=info_text, font_size='14sp', halign='center', 
                         color=(0.8, 0.8, 0.8, 1), text_size=(None, None))
        layout.add_widget(info_label)

        # Estado del modelo
        if MODEL_LOADED:
            status_text = "Modelo de IA cargado correctamente"
            status_color = (0, 1, 0, 1)
        elif PYTORCH_AVAILABLE:
            status_text = "Modelo no encontrado - Modo simulación"
            status_color = (1, 1, 0, 1)
        else:
            status_text = "PyTorch no disponible - Modo simulación"
            status_color = (1, 0.5, 0, 1)
            
        status_label = Label(text=status_text, font_size='12sp', halign='center', 
                           color=status_color)
        layout.add_widget(status_label)

        # Versión
        version_label = Label(text='Version: 1.0.0 (Android)', font_size='14sp', 
                            halign='center', color=(1, 1, 1, 1))
        layout.add_widget(version_label)

        # Creadores
        creators_label = Label(text='Creadores:', font_size='14sp', bold=True,
                             halign='center', color=(1, 1, 1, 1))
        layout.add_widget(creators_label)

        names = ['-Rodrigo Meza', '-Rafael Chui', '-Axel Pariona', '-Liam Quino']
        for name in names:
            name_label = Label(text=name, font_size='12sp', 
                             halign='center', color=(0.8, 0.8, 0.8, 1))
            layout.add_widget(name_label)

        # Botón volver
        btn_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=60)
        btn_layout.add_widget(Widget())
        
        back_btn = Button(text='Volver', size_hint=(None, None), size=(100, 40),
                         background_color=(0.6, 0.1, 0.1, 1))
        back_btn.bind(on_press=self.volver_inicio)
        btn_layout.add_widget(back_btn)
        btn_layout.add_widget(Widget())
        layout.add_widget(btn_layout)

        self.add_widget(layout)

    def volver_inicio(self, instance):
        self.manager.current = 'inicio'

# Clase principal de la aplicación
class DetectorApp(App):
    def build(self):
        """Función principal que construye toda la aplicación"""
        # Crear el gestor de pantallas (permite navegar entre diferentes vistas)
        self.sm = ScreenManager()
        
        # Añadir todas las pantallas de la aplicación
        self.sm.add_widget(SplashScreen(name='splash'))  # Pantalla de carga inicial
        self.sm.add_widget(HomeScreen(name='inicio'))    # Pantalla principal con botones
        self.sm.add_widget(CameraScreen(name='camara'))  # Pantalla de detección en tiempo real
        self.sm.add_widget(ImageScreen(name='imagen'))   # Pantalla de análisis de imagen
        self.sm.add_widget(HelpScreen(name='ayuda'))     # Pantalla de información

        # Programar cambio automático a pantalla principal después de 5 segundos
        Clock.schedule_once(self.ir_a_inicio, 4)
        return self.sm

    def ir_a_inicio(self, dt):
        """Cambiar de la pantalla de carga a la pantalla principal"""
        self.sm.current = 'inicio'

    def on_stop(self):
        """Función que se ejecuta al cerrar la aplicación - limpia recursos"""
        try:
            # Liberar la cámara si está en uso para evitar problemas
            camera_screen = self.sm.get_screen('camara')
            if hasattr(camera_screen, 'capture') and camera_screen.capture is not None:
                camera_screen.capture.release()
        except Exception as e:
            print(f"Error al cerrar: {e}")

# Punto de entrada del programa
if __name__ == '__main__':
    DetectorApp().run()  # Iniciar la aplicación