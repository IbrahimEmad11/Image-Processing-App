import sys
import cv2
from PyQt5.QtWidgets import ( QMainWindow, QApplication, QFileDialog)
from PyQt5.QtGui import QPixmap
from PIL import Image
from task1 import Ui_MainWindow
from task1 import Ui_MainWindow
from image_processing import filters, edge_detection, hp_lp_filters, histogram, threshold_normalization_equalization, hybrid_image


class CVApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initialize_variables()
        self.connect_signals()

    def initialize_variables(self):
        self.input_image = None
        self.input_image2 = None
        self.output_image = None
        self.hybrid_image1 = None
        self.hybrid_image2 = None
        self.gray_img = None
        self.gray_img2 = None
        self.filters_slider = None
        self.lpf_image = None
        self.hpf_image = None
        self.pf_hybrid_flag = True

    def connect_signals(self):
        self.ui.BrowseButton.clicked.connect(self.browse_img)
        self.ui.BrowseButton_2.clicked.connect(self.browse_input_image2)
        self.ui.HighpassButton.clicked.connect(lambda: hp_lp_filters.high_pass_filter(self.gray_img, self))
        self.ui.LowpassButton.clicked.connect(lambda: hp_lp_filters.low_pass_filter(self.gray_img, self))
        self.ui.GenerateHybrid.clicked.connect(lambda: hybrid_image.generate_hybrid_image(self.gray_img, self.gray_img2, self))
        self.ui.horizontalSlider.valueChanged.connect(self.set_cutoff_freq_value)
        self.ui.VerticalSlider.valueChanged.connect(self.set_cutoff_freq_value)
        self.ui.RefreshButton.clicked.connect(self.refresh_img)
        self.ui.AverageFilterButton.clicked.connect(lambda: filters.average_filter(self))
        self.ui.GaussianFilterButton.clicked.connect(lambda: filters.gaussian_filter(self))
        self.ui.MedianFilterButton.clicked.connect(lambda: filters.median_filter(self))
        self.ui.UniformNoiseButton.clicked.connect(lambda: filters.add_uniform_noise(self))
        self.ui.SaltPepperNoiseButton.clicked.connect(lambda: filters.add_salt_and_pepper_noise(self))
        self.ui.GaussianNoiseButton.clicked.connect(lambda: filters.add_gaussian_noise(self))
        self.ui.EqualizeButton.clicked.connect(lambda: threshold_normalization_equalization.image_equalization(self))
        self.ui.NormalizeButton.clicked.connect(lambda: threshold_normalization_equalization.image_normalization(self))
        self.ui.GlobalThresholdButton.clicked.connect(lambda: threshold_normalization_equalization.global_threshold(self))
        self.ui.LocalThresholdButton.clicked.connect(lambda: threshold_normalization_equalization.local_threshold(self))
        self.ui.CannyButton.clicked.connect(lambda: edge_detection.canny_edge_detection(self))
        self.ui.SobelButton.clicked.connect(lambda: edge_detection.perform_sobel_edge_detection(self))
        self.ui.RobetButton.clicked.connect(lambda: edge_detection.perform_roberts_edge_detection(self))
        self.ui.PrewittButton.clicked.connect(lambda: edge_detection.perform_prewitt_edge_detection(self))

    def refresh_img(self):
        self.display_image("grayscale_image.jpg", self.ui.filter_outputImage)
        self.display_image("grayscale_image.jpg", self.ui.Threshold_outputImage)
        self.display_image("grayscale_image.jpg", self.ui.EdgeDetection_outputImage)

    def display_image(self, image_path, label):
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap)

    def set_cutoff_freq_value(self):
        self.cutoff_freq_value = int(self.ui.horizontalSlider.value())
        self.ui.label_5.setText(f"{self.cutoff_freq_value}")
        self.cutoff_freq_value_hybrid = int(self.ui.VerticalSlider.value())
        self.ui.label_7.setText(f"{self.cutoff_freq_value_hybrid}")

    def browse_img(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        self.input_image = Image.open(f"{filename}")

        if filename:
            pixmap = QPixmap(filename)
            if not pixmap.isNull():
                self.ui.filter_inputImage.setPixmap(pixmap)
                self.ui.Threshold_inputImage.setPixmap(pixmap)
                self.ui.EdgeDetection_inputImage.setPixmap(pixmap)
                self.ui.pass_inputImage.setPixmap(pixmap)
                self.ui.hybridInputImage1.setPixmap(pixmap)
                
                self.input_image_cv = cv2.imread(filename)
                self.gray_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

                cv2.imwrite("grayscale_image.jpg", self.gray_img)
                pixmap = QPixmap("grayscale_image.jpg")
                self.ui.filter_outputImage.setPixmap(pixmap)
                self.ui.Threshold_outputImage.setPixmap(pixmap)
                self.ui.EdgeDetection_outputImage.setPixmap(pixmap)
                self.ui.pass_outputImage.setPixmap(pixmap)
                self.ui.freqOutputImage1.setPixmap(pixmap)

        histogram.draw_rgb_histogram(self.input_image_cv, self)
        histogram.draw_rgb_disturb_curve(self.input_image_cv, self)

    def browse_input_image2(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        self.input_image2 = Image.open(f"{filename}")

        if filename:
            pixmap = QPixmap(filename)
            if not pixmap.isNull():
                self.ui.hybridInputImage2.setPixmap(pixmap)
                
                self.input_image2_cv = cv2.imread(filename)
                self.gray_img2 = cv2.cvtColor(self.input_image2_cv, cv2.COLOR_BGR2GRAY)

                cv2.imwrite("grayscale_image.jpg", self.gray_img2)
                pixmap = QPixmap("grayscale_image.jpg")
                self.ui.freqOutputImage2.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CVApp()
    window.setWindowTitle("Task 1")
    window.resize(1450, 950)
    window.show()
    sys.exit(app.exec_())
