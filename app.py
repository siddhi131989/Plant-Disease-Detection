{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7033478b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting streamlit\n",
      "  Downloading streamlit-1.32.1-py2.py3-none-any.whl (8.1 MB)\n",
      "     ---------------------------------------- 8.1/8.1 MB 2.3 MB/s eta 0:00:00\n",
      "Collecting altair<6,>=4.0\n",
      "  Downloading altair-5.2.0-py3-none-any.whl (996 kB)\n",
      "     -------------------------------------- 996.9/996.9 kB 1.9 MB/s eta 0:00:00\n",
      "Collecting blinker<2,>=1.0.0\n",
      "  Using cached blinker-1.7.0-py3-none-any.whl (13 kB)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (5.3.0)\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (8.1.3)\n",
      "Requirement already satisfied: numpy<2,>=1.19.3 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (1.23.5)\n",
      "Requirement already satisfied: packaging<24,>=16.8 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (23.0)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (1.5.3)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (9.4.0)\n",
      "Requirement already satisfied: protobuf<5,>=3.20 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (4.22.0)\n",
      "Collecting pyarrow>=7.0\n",
      "  Downloading pyarrow-15.0.1-cp311-cp311-win_amd64.whl (24.8 MB)\n",
      "     ---------------------------------------- 24.8/24.8 MB 2.5 MB/s eta 0:00:00\n",
      "Requirement already satisfied: requests<3,>=2.27 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (2.28.2)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (13.7.0)\n",
      "Collecting tenacity<9,>=8.1.0\n",
      "  Downloading tenacity-8.2.3-py3-none-any.whl (24 kB)\n",
      "Collecting toml<2,>=0.10.1\n",
      "  Downloading toml-0.10.2-py2.py3-none-any.whl (16 kB)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (4.5.0)\n",
      "Collecting gitpython!=3.1.19,<4,>=3.0.7\n",
      "  Downloading GitPython-3.1.42-py3-none-any.whl (195 kB)\n",
      "     -------------------------------------- 195.4/195.4 kB 2.4 MB/s eta 0:00:00\n",
      "Collecting pydeck<1,>=0.8.0b4\n",
      "  Downloading pydeck-0.8.1b0-py2.py3-none-any.whl (4.8 MB)\n",
      "     ---------------------------------------- 4.8/4.8 MB 2.1 MB/s eta 0:00:00\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (6.2)\n",
      "Collecting watchdog>=2.1.5\n",
      "  Downloading watchdog-4.0.0-py3-none-win_amd64.whl (82 kB)\n",
      "     ---------------------------------------- 82.9/82.9 kB 1.2 MB/s eta 0:00:00\n",
      "Requirement already satisfied: jinja2 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from altair<6,>=4.0->streamlit) (3.1.2)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from altair<6,>=4.0->streamlit) (4.17.3)\n",
      "Collecting toolz\n",
      "  Downloading toolz-0.12.1-py3-none-any.whl (56 kB)\n",
      "     ---------------------------------------- 56.1/56.1 kB 2.9 MB/s eta 0:00:00\n",
      "Requirement already satisfied: colorama in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from click<9,>=7.0->streamlit) (0.4.6)\n",
      "Collecting gitdb<5,>=4.0.1\n",
      "  Downloading gitdb-4.0.11-py3-none-any.whl (62 kB)\n",
      "     ---------------------------------------- 62.7/62.7 kB 3.3 MB/s eta 0:00:00\n",
      "Requirement already satisfied: python-dateutil>=2.8.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2022.7.1)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.0.1)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.4)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests<3,>=2.27->streamlit) (1.26.14)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2022.12.7)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.14.0)\n",
      "Collecting smmap<6,>=3.0.1\n",
      "  Downloading smmap-5.0.1-py3-none-any.whl (24 kB)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.2)\n",
      "Requirement already satisfied: attrs>=17.4.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (22.2.0)\n",
      "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.19.3)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from python-dateutil>=2.8.1->pandas<3,>=1.3.0->streamlit) (1.16.0)\n",
      "Installing collected packages: watchdog, toolz, toml, tenacity, smmap, pyarrow, blinker, pydeck, gitdb, gitpython, altair, streamlit\n",
      "Successfully installed altair-5.2.0 blinker-1.7.0 gitdb-4.0.11 gitpython-3.1.42 pyarrow-15.0.1 pydeck-0.8.1b0 smmap-5.0.1 streamlit-1.32.1 tenacity-8.2.3 toml-0.10.2 toolz-0.12.1 watchdog-4.0.0\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "[notice] A new release of pip available: 22.3.1 -> 24.0\n",
      "[notice] To update, run: C:\\Users\\siddh\\AppData\\Local\\Programs\\Python\\Python311\\python.exe -m pip install --upgrade pip\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "52f12c19",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import tensorflow as tf\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "d132d24d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define paths to the model and label files\n",
    "MODEL_PATH = \"/Users/siddh/OneDrive/Desktop/4th yr project/model/keras_new_model.h5\"\n",
    "LABEL_PATH = \"/Users/siddh/OneDrive/Desktop/4th yr project/model/labelss.txt\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "cb6c1b30",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the Keras model and labels\n",
    "def load_model_and_labels():\n",
    "    model = tf.keras.models.load_model(MODEL_PATH)\n",
    "    with open(LABEL_PATH, 'r') as f:\n",
    "        labels = f.read().splitlines()\n",
    "    return model, labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "33ce3882",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to preprocess the image\n",
    "def preprocess_image(image):\n",
    "    image = tf.image.resize(image, (224, 224))\n",
    "    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)\n",
    "    return image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "f3d8dac5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to make predictions\n",
    "def predict(image, model, labels):\n",
    "    image = np.expand_dims(image, axis=0)\n",
    "    prediction = model.predict(image)\n",
    "    predicted_label = labels[np.argmax(prediction)]\n",
    "    confidence = np.max(prediction)\n",
    "    return predicted_label, confidence"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "381c9251",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Main function to run the Streamlit app\n",
    "def main():\n",
    "    st.title(\"Image Classification\")\n",
    "\n",
    "    # Load model and labels\n",
    "    model, labels = load_model_and_labels()\n",
    "\n",
    "    # Sidebar for file upload\n",
    "    uploaded_file = st.file_uploader(\"Choose an image...\", type=[\"jpg\", \"png\", \"jpeg\"])\n",
    "\n",
    "    # Perform prediction if file uploaded\n",
    "    if uploaded_file is not None:\n",
    "        image = tf.image.decode_image(uploaded_file.read(), channels=3)\n",
    "        st.image(image, caption=\"Uploaded Image\", use_column_width=True)\n",
    "\n",
    "        # Preprocess image\n",
    "        preprocessed_image = preprocess_image(image)\n",
    "\n",
    "        # Make prediction\n",
    "        predicted_label, confidence = predict(preprocessed_image, model, labels)\n",
    "\n",
    "        st.write(\"Prediction:\", predicted_label)\n",
    "        st.write(\"Confidence:\", confidence)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "6a1339ad",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.\n"
     ]
    }
   ],
   "source": [
    "# Run the main function\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "e469a907",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: streamlit in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (1.32.1)\n",
      "Requirement already satisfied: tensorflow in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (2.12.0rc1)\n",
      "Requirement already satisfied: altair<6,>=4.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (5.2.0)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (1.7.0)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (5.3.0)\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (8.1.3)\n",
      "Requirement already satisfied: numpy<2,>=1.19.3 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (1.23.5)\n",
      "Requirement already satisfied: packaging<24,>=16.8 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (23.0)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (1.5.3)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (9.4.0)\n",
      "Requirement already satisfied: protobuf<5,>=3.20 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (4.22.0)\n",
      "Requirement already satisfied: pyarrow>=7.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (15.0.1)\n",
      "Requirement already satisfied: requests<3,>=2.27 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (2.28.2)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (13.7.0)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (8.2.3)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (4.5.0)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (3.1.42)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (0.8.1b0)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (6.2)\n",
      "Requirement already satisfied: watchdog>=2.1.5 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from streamlit) (4.0.0)\n",
      "Requirement already satisfied: tensorflow-intel==2.12.0-rc1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow) (2.12.0rc1)\n",
      "Requirement already satisfied: absl-py>=1.0.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (1.4.0)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (1.6.3)\n",
      "Requirement already satisfied: flatbuffers>=2.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (23.5.26)\n",
      "Requirement already satisfied: gast<=0.4.0,>=0.2.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (0.4.0)\n",
      "Requirement already satisfied: google-pasta>=0.1.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (0.2.0)\n",
      "Requirement already satisfied: h5py>=2.9.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (3.8.0)\n",
      "Requirement already satisfied: jax>=0.3.15 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (0.4.6)\n",
      "Requirement already satisfied: libclang>=13.0.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (15.0.6.1)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (3.3.0)\n",
      "Requirement already satisfied: setuptools in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (65.5.0)\n",
      "Requirement already satisfied: six>=1.12.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (1.16.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (2.2.0)\n",
      "Requirement already satisfied: wrapt<1.15,>=1.11.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (1.14.1)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (1.51.3)\n",
      "Requirement already satisfied: tensorboard<2.13,>=2.12 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (2.12.0)\n",
      "Requirement already satisfied: tensorflow-estimator<2.13,>=2.12.0rc0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (2.12.0)\n",
      "Requirement already satisfied: keras<2.13,>=2.12.0rc0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (2.12.0)\n",
      "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorflow-intel==2.12.0-rc1->tensorflow) (0.30.0)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from altair<6,>=4.0->streamlit) (3.1.2)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from altair<6,>=4.0->streamlit) (4.17.3)\n",
      "Requirement already satisfied: toolz in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
      "Requirement already satisfied: colorama in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from click<9,>=7.0->streamlit) (0.4.6)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
      "Requirement already satisfied: python-dateutil>=2.8.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2022.7.1)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.0.1)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.4)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests<3,>=2.27->streamlit) (1.26.14)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2022.12.7)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.14.0)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.12.0-rc1->tensorflow) (0.38.4)\n",
      "Requirement already satisfied: smmap<6,>=3.0.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
      "Requirement already satisfied: scipy>=1.5 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from jax>=0.3.15->tensorflow-intel==2.12.0-rc1->tensorflow) (1.10.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.2)\n",
      "Requirement already satisfied: attrs>=17.4.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (22.2.0)\n",
      "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.19.3)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
      "Requirement already satisfied: google-auth<3,>=1.6.3 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0-rc1->tensorflow) (2.16.2)\n",
      "Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0-rc1->tensorflow) (0.4.6)\n",
      "Requirement already satisfied: markdown>=2.6.8 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0-rc1->tensorflow) (3.4.1)\n",
      "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0-rc1->tensorflow) (0.7.0)\n",
      "Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0-rc1->tensorflow) (1.8.1)\n",
      "Requirement already satisfied: werkzeug>=1.0.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0-rc1->tensorflow) (2.2.3)\n",
      "Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0-rc1->tensorflow) (0.2.8)\n",
      "Requirement already satisfied: rsa<5,>=3.1.4 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0-rc1->tensorflow) (4.9)\n",
      "Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0-rc1->tensorflow) (1.3.1)\n",
      "Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0-rc1->tensorflow) (0.4.8)\n",
      "Requirement already satisfied: oauthlib>=3.0.0 in c:\\users\\siddh\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0-rc1->tensorflow) (3.2.2)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "[notice] A new release of pip available: 22.3.1 -> 24.0\n",
      "[notice] To update, run: C:\\Users\\siddh\\AppData\\Local\\Programs\\Python\\Python311\\python.exe -m pip install --upgrade pip\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit tensorflow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "9513cb4c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
