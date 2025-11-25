import tensorflow as tf
import streamlit as st

#RECREATE CUSTOM ATTENTION LAYER

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        score = self.score_dense(inputs)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

#LOAD MODEL

MODEL_PATH = r"C:\Users\Naveen Pratap Singh\Documents\Credit Complaints ML\severity_model.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"AttentionLayer": AttentionLayer}
)

labels = ["Low", "Medium", "High"]
colors = ["green", "orange", "red"]  # Define colors for each severity


#STREAMLIT UI
st.title("Complaint Severity Classifier")
st.write("Enter a complaint narrative to predict severity:")

user_text = st.text_area("Complaint Text")

if st.button("Predict Severity"):
    if not user_text.strip():
        st.warning("Please enter a valid complaint.")
    else:
        # Convert string to tensor with batch dimension
        input_tensor = tf.constant([user_text])

        # Predict
        pred = model.predict(input_tensor)
        class_id = int(pred.argmax())

        severity_label = labels[class_id]
        severity_color = colors[class_id]

        # Display colored severity
        st.markdown(f"**Predicted Severity:** <span style='color:{severity_color}'>{severity_label}</span>", unsafe_allow_html=True)
        st.write("Raw model output:", pred)




