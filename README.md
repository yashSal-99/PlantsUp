# 🌿 PlantsUp - A Deep Learning Model to Identify Plants by Photos  

### Ever wondered what the name of that beautiful plant is while exploring nature? 🌱 **PlantsUp** is here to help!  
**PlantsUp** is a Deep Learning-powered solution that can identify the name of a plant or leaf from a photo.  

---

## 🚀 Features  
- **Identify plants and leaves:** Upload a photo of a plant or leaf, and PlantsUp will recognize its name.  
- **Cutting-edge model:** Built on the state-of-the-art **EfficientNet-B3** architecture, ensuring high accuracy and efficiency.  
- **Robust dataset:** Trained on a large collection of **13,000+ images** of plants and leaves.  

---

## 🧠 Model  
PlantsUp uses the **EfficientNet-B3** model, which is:  
- Optimized for high performance and computational efficiency.  
- Ideal for datasets with diverse features like leaf structures and plant textures.  
- Trained end-to-end for precise predictions.  

---

## 📂 Dataset  
The dataset used in this project was sourced from **Kaggle**:  
[Indian Medicinal Leaves Dataset](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset)  
- Contains **13,000+ images** categorized into two classes: **Leaf** and **Plant**.  
- For better generalization, these classes were merged into a single **Merged_Dataset** to allow identification from both types of images.  

---

## 🛠️ Workflows  

### 🔧 **Planthub Notebook**  
This notebook:  
- Merges the dataset directories (Leaf and Plant) into one.  
- Cleans up impurities like non-JPG files, duplicate images, and more.  
- Prepares a clean dataset for training.  

### 🧪 **PlantsUp Notebook**  
This notebook demonstrates the complete machine learning workflow:  
1. **Data Preprocessing**: Ensures the dataset is ready for training.  
2. **Model Selection**: Leverages **EfficientNet-B3** for feature learning.  
3. **Model Training**: Trains the model using the merged dataset.  
4. **Model Evaluation**: Validates performance on unseen data.  
5. **Fine-Tuning**: Optimizes model for better accuracy.  
6. **Model Saving**: Stores the trained model for deployment.  
7. **Model Deployment**: Prepares the model for integration into an app.  

---

## 📱 User Interface  
PlantsUp's user interface is built using **KivyMD** and **Kivy**, popular Python frameworks for app development.  
- **Seamless interaction**: Upload a photo and get instant results.  
- **Mobile-friendly**: The app is optimized for smartphone use.  

---

## 📸 Demo  
![PlantsUp Demo](https://github.com/yashSal-99/PlantsUp/blob/main/Demo1.mp4)  
_See PlantsUp in action!_

