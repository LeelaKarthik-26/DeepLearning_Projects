# 🖼️ CIFAR-10 Image Classification with CNNs  

This project demonstrates image classification on the **CIFAR-10 dataset** using **Convolutional Neural Networks (CNNs)** built with TensorFlow and Keras.  

It includes two models:  
1. **Baseline Model** – a simple CNN architecture.  
2. **Improved Model** – a deeper CNN with **Batch Normalization**, **Dropout**, **Data Augmentation**, and **Learning Rate Scheduling** to improve performance.  

---

## 📂 Dataset  
The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) contains **60,000 32x32 color images** in **10 classes**:  

- airplane ✈️  
- automobile 🚗  
- bird 🐦  
- cat 🐱  
- deer 🦌  
- dog 🐶  
- frog 🐸  
- horse 🐴  
- ship 🚢  
- truck 🚚  

**Split:**  
- 50,000 training images  
- 10,000 testing images  

---

## 🧠 Models

### 🔹 Baseline Model
- 2 Convolutional layers (32, 64 filters)  
- MaxPooling layers  
- Dense layer with 128 units  
- Dropout (0.5)  
- Softmax output layer  

**Test Accuracy (Baseline):** ~0.68  

---

### 🔹 Improved Model
- Convolutional blocks with 32 → 64 → 128 filters  
- **Batch Normalization** for stable training  
- **Dropout** for regularization  
- **Data Augmentation** (rotation, shift, horizontal flip)  
- **Learning Rate Scheduling** with `ReduceLROnPlateau`  
- **Early Stopping** for better generalization  

**Test Accuracy (Improved):** ~0.82  

---

## ⚙️ Training Setup
- Optimizer: **Adam (lr = 0.001 with scheduling)**  
- Loss: **Categorical Crossentropy**  
- Batch Size: **64**  
- Epochs: **50 (with early stopping)**  
- Validation Split: **20%**  

---

## 📊 Results

### 🔸 Baseline Model
- **Training Accuracy:** ~85%  
- **Validation Accuracy:** ~68%  
- **Test Accuracy:** ~68%  
- **Observation:** Model overfits quickly and struggles with generalization.  

### 🔸 Improved Model
- **Training Accuracy:** ~95%  
- **Validation Accuracy:** ~82%  
- **Test Accuracy:** ~82%  
- **Observation:** Regularization + data augmentation improves robustness and reduces overfitting.  

---

### 📈 Accuracy Curves
![Accuracy Curve](results/accuracy_curve.png)

### 📉 Loss Curves
![Loss Curve](results/loss_curve.png)

> *(You can generate and save these plots using Matplotlib from the training script.)*  

---

## 🚀 Future Improvements
- Apply **Transfer Learning** (ResNet50, MobileNetV2, EfficientNet).  
- Hyperparameter tuning (learning rate, batch size, kernel size).  
- Add **regularization** (L2 weight decay).  
- Deploy as a **web app** (Flask / Streamlit).  

---

## 🛠️ Tech Stack
- Python 🐍  
- TensorFlow / Keras  
- NumPy, Matplotlib  
- Jupyter / Colab (optional)  

---

## 📌 How to Run
```bash
# Clone the repository
git clone https://github.com/your-username/cifar10-cnn.git
cd cifar10-cnn

# Install dependencies
pip install -r requirements.txt

# Run training script
python cifar10_cnn.py
