# Healthcare Project

A comprehensive healthcare-focused machine learning and AI toolkit designed to assist with disease prediction, symptom analysis, chatbot interaction, and COVID-19 detection.

## Features

- **Heart Disease Prediction:** Uses trained machine learning models (XGBoost, SVM, neural networks) to predict the likelihood of heart disease based on user input or dataset samples.
- **Symptom Model:** Processes user symptoms and predicts possible health conditions, utilizing pretrained tokenizers.
- **Healthcare Chatbot:** Interacts with users, answers health-related questions, and provides guidance based on intents and classes stored in the chatbot folder.
- **COVID-19 Detection:** Predicts COVID-19, normal, or viral pneumonia conditions from medical images using a deep learning model.

## Directory Structure

- `heart_diseas/knowledgeBaseHeartDisease.ipynb`: Main notebook for heart disease data analysis and model training.
- `symtom/`: Contains symptom analysis models and tokenizers.
- `Chatbot/`: Holds files for chatbot functionality (`intents.json`, `words.pkl`, `classes.pkl`, `chatbotmodel.h5`).
- `Covid 19/`: Stores image detection models for COVID-19 (`detection_model.h5`).
- `all_in_colab.ipynb`: Colab notebook combining all core functions and demonstrating usage.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/El-harakani27/Healthcare-project.git
   cd Healthcare-project
   ```

2. **Dependencies:**
   - Python 3.x
   - Jupyter Notebook/Google Colab
   - Required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost`, `tensorflow`, `nltk`, `pickle`

   Install with:
   ```bash
   pip install numpy pandas matplotlib scikit-learn xgboost tensorflow nltk
   ```

   For NLTK, download required datasets inside your notebook:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

3. **Pretrained Models and Data Files:**  
   Upload or place the following files in their respective folders:
   - Chatbot: `intents.json`, `words.pkl`, `classes.pkl`, `chatbotmodel.h5`
   - Symptom: `mohamed_tokenizer.pickle`
   - Heart Disease: `model_heart.pkl`
   - COVID-19 Detection: `detection_model.h5`
   - Sample medical images for COVID-19 prediction

4. **Run Notebooks:**  
   Open `all_in_colab.ipynb` or `heart_diseas/knowledgeBaseHeartDisease.ipynb` in Jupyter or Google Colab and follow the instructions in the markdown cells to load models and test functionalities.

## Example Usage

- **Heart Disease Prediction:**
  - Input patient data in the notebook code cell or use sample arrays provided.
  - Execute prediction cell to get results.

- **Chatbot:**
  - Load chatbot files as instructed and interact via the provided input loop.

- **COVID-19 Detection:**
  - Load the detection model.
  - Use the `predict_transfer_detect` function with image paths to classify images.

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.

## License

This repository currently does not specify a license.

## Author

Developed by [El-harakani27](https://github.com/El-harakani27)
