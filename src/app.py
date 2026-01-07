from pickle import load
import streamlit as st

model = load(open("src/models/random-forest-diabetes.pkl","rb"))
class_dict = {"0":"No sufre de diabetes",
                "1":"Sufre de diabetes"}

st.title("Diabetes - Model prediction")
st.markdown("""Power by: [Rodolfo D`alessandro (Elreno23)] (https://github.com/Elreno23)""")
st.divider()


pregnancies = st.number_input(
    "Enter the number of pregnancies", value=0, min_value=0, max_value=17, placeholder="Type a integer number...")

glucose = st.number_input(
    "Enter the glucose level", value=40, min_value=40, max_value=400, placeholder="Type a integer number...")

bmi = st.number_input(
    "Enter BMI value", value=15.0, min_value=15.0, max_value=60.0, placeholder="Type a number with decimals...")

diabetes_pedigree_function = st.number_input(
    "Enter Hereditary Diabetes Risk (DPF) value", value=0.084, min_value=0.084, max_value=1.300, placeholder="Type a number with decimals...")

age = st.number_input(
    "Enter your age", value=12, min_value=12, max_value=80, placeholder="Type a integer number...")

if st.button("Predict"):
    prediction = str(model.predict([[pregnancies,glucose,bmi,diabetes_pedigree_function,age]])[0])
    predict_class = class_dict[prediction]

    st.divider()
    st.write("Prediction:", predict_class)
    st.divider()
