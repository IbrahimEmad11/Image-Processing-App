##########################################################################################################################################
    def add_uniform_noise_opencv(self):
        noise = np.random.uniform(low=-50, high=50, size=self.input_image.shape).astype(np.uint8)  # Generate uniform noise
        noisy_image = cv2.add(self.input_image, noise)

        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        height, width, channel = noisy_image.shape
        bytesPerLine = 3 * width
        qImg = QImage(noisy_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.filter_outputImage.setPixmap(pixmap)

    def add_gaussian_noise_opencv(self):
        noise = np.random.normal(loc=0, scale=50, size=self.input_image.shape).astype(np.uint8)  # Generate Gaussian noise
        noisy_image = cv2.add(self.input_image, noise)

        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        height, width, channel = noisy_image.shape
        bytesPerLine = 3 * width
        qImg = QImage(noisy_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.filter_outputImage.setPixmap(pixmap)

    def add_salt_and_pepper_noise_opencv(self, amount=0.05):
        noisy_image = np.copy(self.input_image)

        # Generate salt and pepper noise mask
        num_salt = np.ceil(amount * self.input_image.size * 0.5)
        salt_coords = [np.random.randint(0, i, int(num_salt)) for i in self.input_image.shape[:-1]]  
        noisy_image[tuple(salt_coords)] = 255

        num_pepper = np.ceil(amount * self.input_image.size * 0.5)
        pepper_coords = [np.random.randint(0, i, int(num_pepper)) for i in self.input_image.shape[:-1]]  
        noisy_image[tuple(pepper_coords)] = 0

        # Convert to QImage and display
        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        height, width, channel = noisy_image.shape
        bytesPerLine = 3 * width
        qImg = QImage(noisy_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.filter_outputImage.setPixmap(pixmap)

    def average_filter_opencv(self, kernel_size=3):
        if self.ui.Radio3x3Kernal.isChecked():
            kernel_size = 3
        elif self.ui.Radio5x5Kernal.isChecked():
            kernel_size = 5

        # Define the filter kernel
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

        # Apply the filter using convolution
        filtered_image = cv2.filter2D(self.input_image, -1, kernel)

        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
        height, width, channel = filtered_image.shape
        bytesPerLine = 3 * width
        qImg = QImage(filtered_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.filter_outputImage.setPixmap(pixmap)

    def gaussian_filter_opencv(self, kernel_size=3, sigma=1):
        if self.ui.Radio3x3Kernal.isChecked():
            kernel_size = 3
        elif self.ui.Radio5x5Kernal.isChecked():
            kernel_size = 5

        # Generate Gaussian kernel
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = np.outer(kernel, kernel.transpose())

        # Apply the filter using convolution
        filtered_image = cv2.filter2D(self.input_image, -1, kernel)

        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
        height, width, channel = filtered_image.shape
        bytesPerLine = 3 * width
        qImg = QImage(filtered_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.filter_outputImage.setPixmap(pixmap)

    def apply_median_filter_opencv(self, size=3):
        if self.ui.Radio3x3Kernal.isChecked():
            kernel_size = 3
        elif self.ui.Radio5x5Kernal.isChecked():
            kernel_size = 5
            
        filtered_image = cv2.medianBlur(self.input_image, kernel_size)
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
        height, width, channel = filtered_image.shape
        bytesPerLine = 3 * width
        qImg = QImage(filtered_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.filter_outputImage.setPixmap(pixmap)
    ###########################################################################################################################################