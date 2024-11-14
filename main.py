import joblib
import spacy
from spacy.tokens import Doc
import streamlit as st
import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt

# โหลดโมเดล
try:
    model = joblib.load("model.joblib")
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    model = None  # กำหนดให้ model เป็น None หากการโหลดล้มเหลว

stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

def tokens_to_features(tokens, i):
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,
        "word.isdigit()": word.isdigit(),
        "word.islen5": word.isdigit() and len(word) == 5
    }

    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace(),
            "-1.word.is_stopword()": prevword in stopwords,
            "-1.word.isdigit()": prevword.isdigit(),
        })
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace(),
            "+1.word.is_stopword()": nextword in stopwords,
            "+1.word.isdigit()": nextword.isdigit(),
        })
    else:
        features["EOS"] = True

    return features



def parse_and_visualize(text):
    try:
        tokens = text.split()
        features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
        
        # ตรวจสอบว่าโมเดลถูกโหลดก่อนการทำนาย
        if model is None:
            raise ValueError("โมเดลยังไม่ได้ถูกโหลด.")
        
        predictions = model.predict([features])[0]

        # ใช้ SpaCy Thai tokenizer
        nlp = spacy.blank("th")  # ตรวจสอบให้แน่ใจว่าคุณมี Thai tokenizer ที่เหมาะสม
        doc = Doc(nlp.vocab, words=tokens)

        # กำหนดสีที่จะแสดงตาม label
        label_colors = {
            "O": "#b084c9",  # Pastel purple for non-entity (O)
            "ADDR": "#ffcc66",  # Pastel orange for ADDA
            "LOC": "#8ab6cb",  # Pastel light blue for LOC
            "POST": "#ff6666"  # Pastel red for POST
        }

        # สร้าง HTML output โดยคำและป้ายกำกับอยู่ในกล่องเดียวกัน
        html_output = '<div style="font-family: sans-serif; text-align: left; line-height: 1.5;">'
        for i, token in enumerate(tokens):
            label = predictions[i]
            label_color = label_colors.get(label, "#ffffff")  # Default to white if no color is assigned

            # สร้างกล่องสำหรับคำและป้ายกำกับ
            if label != ' ':  # ตรวจสอบว่ามีป้ายกำกับหรือไม่
                html_output += f'<div style="display: inline-block; margin: 0 5px; text-align: center; border: 1px solid {label_color}; background-color: {label_color}; padding: 5px; border-radius: 5px;">'
                html_output += f'<div style="background-color: white; color: black; padding: 2px 5px; margin-top: 2px; border-radius: 3px;">{token}</div>'
                html_output += f'<div style="background-color: {label_color}; color: white; padding: 2px 5px; margin-top: 2px; border-radius: 3px; font-weight: bold; font-size: 0.8em;">{label}</div>'
                html_output += '</div>'
            else:
                html_output += f'<div style="display: inline-block; margin: 0 5px; text-align: center; border: 0px solid #007bff; padding: 5px; border-radius: 5px;">'
                html_output += f'<div>{token}</div>'
                html_output += f'<div style="background-color: #ffffff; color: white; padding: 2px 5px; margin-top: 2px; border-radius: 3px; font-weight: bold; font-size: 0.8em;">{label}</div>'
                html_output += '</div>'
        html_output += '</div>'
        
        return html_output

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผล NER: {e}")
        return ""

def parse(text):
  tokens = text.split()
  features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
  return model.predict([features])[0]

# Convert the Counter to a DataFrame
def create_dataframe_result(data):
    df_result_counter = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df_result_counter.columns = ['Class', 'Count']
    return df_result_counter


# แอปพลิเคชัน Streamlit
# Title
st.title("Named Entity Recognition (NER)")

# Text area input for user
text_input = st.text_area("กรอกข้อความที่นี่:", "")

# Initialize session states for NER output and button visibility
if 'ner_output' not in st.session_state:
    st.session_state.ner_output = None
if 'show_shuffle' not in st.session_state:
    st.session_state.show_shuffle = False

# Create a container with two columns
col1, col2 = st.columns([1, 1])

# Create "Analyst" button in the first column
with col1:
    button_analyst = st.button("Analyst")

# Check if "Analyst" button is pressed
if button_analyst:
    if text_input:
        # Simulate NER parsing and visualization
        html_output = parse_and_visualize(text_input)
        if html_output:
            st.session_state.ner_output = html_output  # Store output in session state
            st.markdown(html_output, unsafe_allow_html=True)
            st.session_state.show_shuffle = True  # Show "Shuffle Sentences" button
    else:
        st.warning("กรุณากรอกข้อความเพื่อวิเคราะห์.")

# Show "Shuffle Sentences" button right next to "Analyst" if "Analyst" was pressed
if st.session_state.show_shuffle:
    with col1:
        button_shuffle = st.button("Shuffle Sentences")  # Placed next to "Analyst"
    if button_shuffle:
        if text_input:
            # Display the NER output if it exists
            if 'ner_output' in st.session_state and st.session_state.ner_output:
                st.markdown(st.session_state.ner_output, unsafe_allow_html=True)
            for i in range(5):
                words = text_input.split()  # Split text into words
                random.shuffle(words)  # Shuffle word order
                shuffled_text = " ".join(words)  # Join words back into text
                result = parse(shuffled_text)  # Assuming parse returns a list of entities

                # Display the shuffled text below the NER output
                st.write(f"Shuffled Text {i + 1}:")
                st.write(shuffled_text)  # Display the shuffled text

                # Convert result to DataFrame and display as a smaller bar chart
                df = create_dataframe_result(Counter(result))
                st.write(df)
                fig, ax = plt.subplots(figsize=(5, 3))  # Adjust size here (width=5, height=3)
                ax.bar(df['Class'], df['Count'])
                ax.set_xlabel("Entity Class")
                ax.set_ylabel("Count")
                ax.set_title("Entity Counts")
                st.pyplot(fig)  # Display the smaller bar chart
                st.write("---")
