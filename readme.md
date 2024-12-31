# Flower Classification Project

This project focuses on building a deep learning model for flower classification and providing a user-friendly interface through a Django-based web application.

## Features
- **Deep Learning Model:** A trained ONNX model for classifying flower images into multiple categories using a state-of-the-art deep learning pipeline.
- **Django Web Application:** A user interface that allows users to upload images and receive predictions.
- **Interactive Visualization:** Displays classification results and related metrics.

## Deep Learning Model
The classification model was trained using a comprehensive PyTorch-based pipeline with the following stages:

### 1. Data Collection and Preprocessing
- **Dataset:** The dataset comprises images of flowers categorized into multiple classes such as daisy, dandelion, rose, sunflower, and tulip.
- **Data Augmentation:** Techniques such as horizontal and vertical flipping, rotation, color jitter, and normalization were applied to increase dataset variability and improve model robustness.

### 2. Model Architecture
- **Transfer Learning:** A pre-trained model (e.g., ResNet50, EfficientNet) was fine-tuned to adapt to the flower classification task.
- **Custom Layers:** Additional fully connected layers with dropout were added to prevent overfitting and to specialize in the flower dataset.
- **Frameworks:** PyTorch was used for model development, and the final model was exported to ONNX format for compatibility.

### 3. Training and Optimization
- **Loss Function:** Cross-entropy loss was employed as the objective function.
- **Optimizer:** The Adam optimizer was used with a learning rate scheduler to ensure smooth convergence.
- **Batch Size:** Training was performed with a batch size optimized for both accuracy and GPU memory usage.
- **Metrics:** Accuracy, precision, recall, and F1-score were tracked to evaluate performance during training.

### 4. Evaluation
- **Confusion Matrix:** A confusion matrix was generated to analyze model performance for each class.
- **Validation:** The model was validated on a hold-out dataset to ensure generalization.
- **Results:** The model achieved high accuracy on the test set, demonstrating its effectiveness.

### 5. Deployment
- The trained model was exported to ONNX format (`model.onnx`) for integration with the Django web application.

## Project Structure

Web app directory:
```
flower_classifier/
|-- classifier/
|   |-- migrations/        # Database migrations
|   |-- static/            # Static files (CSS, JS, Images)
|   |-- templates/         # HTML templates for the web app
|   |-- __init__.py        # App initialization
|   |-- admin.py           # Admin interface
|   |-- apps.py            # App configuration
|   |-- models.py          # Database models (if any)
|   |-- views.py           # Views for handling requests
|   |-- urls.py            # URL routing
|-- media/                 # Uploaded media files
|-- db.sqlite3             # SQLite database
|-- manage.py              # Django management script
```

Other directories and files:
```
|-- augmented_train/       # Augmented training dataset
|-- train/                 # Training scripts and data preprocessing
|-- model.onnx             # Trained ONNX model for predictions
|-- requirements.txt       # Python dependencies
|-- confusion_matrix.csv   # Confusion matrix from evaluation
|-- nb.ipynb               # Jupyter Notebook with training pipeline
|-- training_stats.json    # Training statistics
```

## Installation and Setup

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.8 or higher
- pip (Python package installer)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd flower_classifier
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply Migrations**:
   ```bash
   python manage.py migrate
   ```

5. **Run the Development Server**:
   ```bash
   python manage.py runserver
   ```
   The application will be available at `http://127.0.0.1:8000/`.

## Usage
- Open the web application in your browser.
- Upload an image of a flower using the provided interface.
- View the classification result and additional metrics.

## Future Work
- Add more flower categories to improve classification.
- Deploy the application to a cloud platform for public access.
- Implement real-time image classification.

---

Feel free to contribute to this project by submitting pull requests or reporting issues!
