
# **Generating Handwritten Digits with DCGAN**  
This project demonstrates the use of a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic handwritten digits trained on the **MNIST dataset**. It provides an end-to-end implementation, from loading the dataset to training and visualizing the generated results.

---

## **Abstract**  
Generative Adversarial Networks (GANs) are a class of neural networks that generate new data similar to a given dataset. This project implements a DCGAN, which uses convolutional layers to generate high-quality images of handwritten digits. The generator creates fake handwritten digits, while the discriminator evaluates their authenticity. Through iterative training, the generator learns to produce increasingly realistic outputs.  

This project is built for beginners exploring GANs and leverages the MNIST dataset to create a practical example of generative modeling.

---

## **How It Works**
1. **Dataset**:  
   The **MNIST dataset** of handwritten digits (0–9) is used. It consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels in grayscale.  

2. **Generator**:  
   - Takes random noise as input and generates fake images resembling handwritten digits.  
   - Built with transposed convolutional layers to upsample noise into image dimensions.  

3. **Discriminator**:  
   - Differentiates between real MNIST images and fake images produced by the generator.  
   - Built with convolutional layers to classify input images as real or fake.  

4. **Training**:  
   - Both models are trained simultaneously in an adversarial setup.  
   - The generator aims to fool the discriminator, while the discriminator aims to correctly identify fake images.  
   - Loss functions and optimizers are applied to improve both models iteratively.  

5. **Visualization**:  
   - During training, generated images are saved after every epoch to monitor progress.  
   - The generator produces high-quality digits after sufficient training.  

---

## **Features**  
- Generates handwritten digits from random noise.  
- Interactive visualizations for monitoring training progress.  
- Supports training in Google Colab with GPU acceleration.  
- Fully customizable architecture for both generator and discriminator.  

---

## **Technologies Used**
- **Python**: Programming language for implementing the DCGAN.  
- **TensorFlow/Keras**: Framework for building and training deep learning models.  
- **Matplotlib**: For visualizing generated results.  
- **Google Colab**: GPU-enabled environment for training.  

---

## **Setup Instructions**
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/DCGAN-MNIST.git
   cd DCGAN-MNIST
   ```  

2. Open the notebook in **Google Colab** for seamless training with GPU support.  

3. Install dependencies (if running locally):  
   ```bash
   pip install tensorflow matplotlib
   ```  

4. Train the model:  
   Run the script to train the DCGAN on the MNIST dataset.  

5. Generate handwritten digits:  
   Use the generator model to produce realistic handwritten-style images.  

---

## **Outputs**
- Generated images saved as `image_at_epoch_XXXX.png` during training.  
- Visualizations of generated handwritten digits after training completion.
- ![download](https://github.com/user-attachments/assets/bb16eb0b-ed7e-4101-aa35-27d0bf9007c6)
![1](https://github.com/user-attachments/assets/b26c0ad0-3000-44c3-a21e-0daf3ebac6b7)
![download1](https://github.com/user-attachments/assets/05351bb7-41de-4331-8eb2-fc7b3f10d6a5)


---

## **References**
- **Research Papers**:  
  - [GANs: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) by Ian Goodfellow et al.  
  - [Unsupervised Representation Learning with Deep Convolutional GANs](https://arxiv.org/abs/1511.06434) by Alec Radford et al.  

- **MNIST Dataset**:  
  - [MNIST Official Website](http://yann.lecun.com/exdb/mnist/)  

- **TensorFlow Documentation**:  
  - [TensorFlow GAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)  

---

## **Future Work**
- Implement **Conditional GAN (CGAN)** for generating specific digits based on labels.  
- Explore larger and more diverse datasets, such as CIFAR-10 or CelebA.  
- Enhance training speed and stability by using advanced techniques like gradient penalty or spectral normalization.  

---

## **Acknowledgments**
This project is inspired by the TensorFlow DCGAN tutorial and aims to provide a beginner-friendly implementation for learning generative modeling concepts.  

