# LVDCS: Minimising Doubts in Distance Learning

This repository contains the source code for our Capstone Project implemented in our final year of engineering. LVDCS stands for Lecture Video Doubt Clarification System wherein users can get answers immediately for any doubts they have while watching lecture videos. Below depicts the workflow of the entire process.

![workflow](./images/final_pipeline.png)

Table of Contents
====
+ [Installation](#installation)
+ [Usage](#usage)



Installation
===

### 1. Clone the repository
```sh
git clone https://github.com/anaghapatill/LVDCS.git
```

### 2. Navigate to the *LVDCS-main* folder
```sh
cd LVDCS-main
```

### 3. Install the required Python libraries
```sh 
pip install -r requirements.txt
```

Usage
===


### 1. Add the AssemblyAI API key in the `main.py` file and the HuggingFace API token in the `upload_llm.py` and the `link_llm.py` files
```python
api_key = "<AssemblyAI API Key>"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<HuggingFace API Token>"
```

### 2. Run the `main.py` file

``` python
streamlit run main.py
```
<br />

## **Using the Lecture Video Doubt Clarification System**
<p></p>

### 1. Upload video(s) from your local system using the `Upload Local Video(s)` option on the Streamit page

![upload](./images/upload.png)

<p></p>

### 2. Add YouTube link(s) of any video(s) using the `Add YouTube Video Link(s)` option on the Streamit page

![yt](./images/yt.png)

<p>

**Important: Users must ensure that the videos must be in the English language and YouTube videos must contain closed captions in English for the system to work efficiently.**
