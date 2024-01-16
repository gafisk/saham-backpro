import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from ast import literal_eval
import locale


# Function to load dataset
@st.cache(allow_output_mutation=True)
def load_dataset(uploaded_file):
    return pd.read_excel(uploaded_file)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("loss") is not None and logs.get("loss") < 0.0001:
            print("\nTraining dihentikan, Error sudah mencapai 0.00001")
            self.model.stop_training = True


# Set session state variables
if "dataset" not in st.session_state:
    st.session_state.dataset = None
    st.session_state.fitur = None
    st.session_state.target = None
    st.session_state.dataX = None
    st.session_state.dataY = None
    st.session_state.normalisasiX = None
    st.session_state.normalisasiY = None
    st.session_state.layers = [0, 0, 1]
    st.session_state.learningRate = 0
    st.session_state.epochs = 0
    st.session_state.y_test = None
    st.session_state.pred = None
    st.session_state.scaler = MinMaxScaler()
    st.session_state.model = None

# Variabel
# scaler = MinMaxScaler()

# Layout with columns
col1 = st.sidebar
col2 = st

col1.title("Menu Bar")

menu_options = [
    "Dashboard",
    "Upload File",
    "Set Target Fitur",
    "Normalisasi",
    "Training",
    "Evaluasi Model",
    "Prediksi",
    "Prediksi Best Model",
]

selected_option = col1.radio("Pilih Menu", menu_options)

col2.title(f"Menu {selected_option}")

if selected_option == "Dashboard":
    col2.write("Welcome to the Home page!")

if selected_option == "Upload File":
    uploaded_file = col2.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    if uploaded_file is not None:
        st.session_state.dataset = load_dataset(uploaded_file)
        st.session_state.fitur = (
            None  # Reset selected columns when new file is uploaded
        )
        st.session_state.target = None
    if st.session_state.dataset is not None:
        col2.write("This is the Profile page. Using the dataset:")
        col2.dataframe(st.session_state.dataset, width=800)
    else:
        col2.write("Mohon Upload Data Dulu.")

if selected_option == "Set Target Fitur":
    col2.write("Pilih Fitur dan Target:")
    if st.session_state.dataset is not None:
        col2a, col2b = col2.columns([1, 1])

        # Multiselect for features
        selected_columns_X = col2a.multiselect(
            "Pilih Fitur",
            st.session_state.dataset.columns,
            st.session_state.fitur,
        )

        # Button to update features
        if col2a.button("Tambah Fitur"):
            st.session_state.fitur = selected_columns_X
            st.session_state.dataX = st.session_state.dataset[selected_columns_X]

        col2a.dataframe(st.session_state.dataX, width=400)

        # Multiselect for target
        selected_columns_Y = col2b.multiselect(
            "Pilih Target",
            st.session_state.dataset.columns,
            st.session_state.target,
        )

        # Button to update target
        if col2b.button("Tambah Target"):
            st.session_state.target = selected_columns_Y
            st.session_state.dataY = st.session_state.dataset[selected_columns_Y]

        col2b.dataframe(st.session_state.dataY, width=400)
    else:
        col2.write("Mohon Upload Data Dulu.")

if selected_option == "Normalisasi":
    if st.session_state.dataX is not None:
        col2a, col2b = col2.columns([1, 1])
        col2a.write("Data Fitur Asli")
        col2a.dataframe(st.session_state.dataX, width=400)
        col2b.write("Data Fitur Hasil Normalisasi")
        st.session_state.normalisasiX = st.session_state.scaler.fit_transform(
            st.session_state.dataX
        )
        normalisasiX = pd.DataFrame(
            st.session_state.normalisasiX, columns=st.session_state.fitur
        )
        col2b.dataframe(normalisasiX, width=400)
    else:
        col2.write("Pilih setidaknya satu kolom fitur pada halaman Set Target Fitur.")

    if st.session_state.dataY is not None:
        col2c, col2d = col2.columns([1, 1])
        col2c.write("Data Fitur Asli")
        col2c.dataframe(st.session_state.dataY, width=400)
        col2d.write("Data Target Hasil Normalisasi")
        st.session_state.normalisasiY = st.session_state.scaler.fit_transform(
            st.session_state.dataY.values.reshape(-1, 1)
        )
        normalisasiY = pd.DataFrame(
            st.session_state.normalisasiY,
            columns=st.session_state.target,
        )
        col2d.dataframe(normalisasiY, width=400)
    else:
        col2.write("Pilih setidaknya satu kolom target pada halaman Set Target Fitur.")

if selected_option == "Training":
    col2.write("Data Fitur Asli")
    col2a, col2b, col2c = col2.columns(3)
    col2d, col2e = col2.columns(2)

    # Input untuk Layer
    layer_input_1 = col2a.text_input("Neuron Input Layer", st.session_state.layers[0])
    st.session_state.layers[0] = int(layer_input_1)

    layer_input_2 = col2b.text_input("Neuron Hidden Layer", st.session_state.layers[1])
    st.session_state.layers[1] = int(layer_input_2)

    layer_input_3 = col2c.text_input("Neuron Output Layer", st.session_state.layers[2])
    st.session_state.layers[2] = int(layer_input_3)

    # # Input untuk learning rate
    learning_rate_input = col2d.text_input(
        "Learning Rate (e.g., 0.001)", st.session_state.learningRate
    )
    st.session_state.learningRate = float(learning_rate_input)

    epoch_input = col2e.text_input("Input Epoch (e.g., 200)", st.session_state.epochs)
    st.session_state.epochs = int(epoch_input)

    # # Input untuk epoch
    if col2.button("Training"):
        if (
            st.session_state.normalisasiY is not None
            and st.session_state.normalisasiX is not None
        ):
            split_index = int(0.7 * len(st.session_state.normalisasiX))
            X_train = st.session_state.normalisasiX[:split_index]
            y_train = st.session_state.normalisasiY[:split_index]
            X_test = st.session_state.normalisasiX[split_index:]
            st.session_state.y_test = st.session_state.normalisasiY[split_index:]
            model = Sequential()
            model.add(
                Dense(
                    st.session_state.layers[1],
                    input_dim=st.session_state.layers[0],
                ),
            )
            model.add(
                Dense(
                    st.session_state.layers[2],
                    activation="sigmoid",
                )
            )
            optimizer = Adam(learning_rate=st.session_state.learningRate)
            model.compile(loss="mean_squared_error", optimizer=optimizer, metrics="mse")
            callbacks = myCallback()
            history = model.fit(
                X_train,
                y_train,
                epochs=st.session_state.epochs,
                verbose=1,
                batch_size=32,
                callbacks=[callbacks],
            )
            st.session_state.model = model
            st.session_state.pred = model.predict(X_test)
            st.write("Training Selesai")
        else:
            st.write("Lakukan Tahapan Sebelumnya Dahulu")

if selected_option == "Evaluasi Model":
    if st.session_state.model is not None:
        y_test = st.session_state.scaler.inverse_transform(st.session_state.y_test)
        y_pred = st.session_state.scaler.inverse_transform(st.session_state.pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        mse = np.mean((st.session_state.y_test - st.session_state.pred) ** 2)
        col2.write("MSE", mse)
        col2.write("NILAI MAPE", mape)
        fig, ax = plt.subplots()
        ax.plot(y_test, label="Nilai Aktual")
        ax.plot(y_pred, label="Nilai Prediksi")
        ax.set_xlabel("Index")
        ax.set_ylabel("Nilai")
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("Training Dulu Modelmu")

if selected_option == "Prediksi":
    if st.session_state.dataY is not None:
        col2a, col2b, col2c, col3e = col2.columns(4)
        model = st.session_state.model
        n_ekspor = col2a.text_input("Nilai Ekspor")
        n_impor = col2b.text_input("Nilai Import")
        n_inflasi = col2c.text_input("Nilai Inflasi")
        n_bunga = col3e.text_input("Nilai Suku Bunga")
        if st.button("Prediksi"):
            if n_ekspor and n_impor and n_inflasi and n_bunga:
                ekspor = float(n_ekspor)
                impor = float(n_impor)
                inflasi = float(n_inflasi)
                bunga = float(n_bunga)
                data = [[ekspor, impor, inflasi, bunga]]
                prediction = model.predict(data)
                st.write(
                    "Hasil Prediksi Nilai Kurs:",
                    round(
                        st.session_state.scaler.inverse_transform(prediction)[-1][-1], 2
                    ),
                )
            else:
                st.write("Masukkan semua nilai yang diperlukan.")
    else:
        st.write("Training dulu Modelmu")

if selected_option == "Prediksi Best Model":
    n_bulan = st.text_input("Jumlah Bulan")
    locale.setlocale(locale.LC_ALL, "id_ID.UTF-8")
    if st.button("Prediksi"):
        split_index = int(0.7 * len(st.session_state.normalisasiX))
        X_trainn = st.session_state.normalisasiX[:split_index]
        y_trainn = st.session_state.normalisasiY[:split_index]
        X_testt = st.session_state.normalisasiX[split_index:]
        y_testt = st.session_state.normalisasiY[split_index:]

        loaded_model = load_model("model.h5")
        predd = loaded_model.predict(X_testt)

        y_inv = st.session_state.scaler.inverse_transform(y_testt)
        y_t = st.session_state.scaler.inverse_transform(y_trainn)
        pred_inv = st.session_state.scaler.inverse_transform(predd)

        bulan = int(n_bulan)

        next_data = X_testt[-1].reshape(1, -1)
        next_predict = []
        for _ in range(bulan):
            next_month_pred = loaded_model.predict(next_data)
            next_predict.append(
                st.session_state.scaler.inverse_transform(next_month_pred)
            )
            next_data = np.concatenate([next_data[:, 1:], next_month_pred], axis=1)
        extended_range = range(len(y_t), len(y_t) + len(y_testt) + bulan)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(range(len(y_t)), y_t, color="blue", label="Training")
        ax.plot(extended_range[:-bulan], y_inv, color="red", label="Testing")
        ax.plot(
            range(len(y_t), len(y_t) + len(y_testt)),
            pred_inv,
            color="orange",
            label="Pred",
        )
        ax.plot(
            range(len(y_t) + len(y_testt), len(y_t) + len(y_testt) + bulan),
            np.concatenate(next_predict),
            color="purple",
            label=f"Prediksi {bulan} Bulan Kedepan)",
        )
        ax.set_xlabel("Jumlah Data")
        ax.set_ylabel("Nilai Kurs")
        ax.set_title(f"Plotting data Training dan Testing")
        ax.legend()
        st.pyplot(fig)
        for i in range(len(next_predict)):
            nilai = int(next_predict[i][0][0])
            formatted_value = locale.currency(nilai, grouping=True, symbol=True)
            st.write(
                f"Hasil Prediksi Kurs pada bulan ke - {i+1} selanjutnya = {formatted_value}"
            )
