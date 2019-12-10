from tensorflow.keras.applications import VGG19

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, UpSampling2D, Lambda

encoder = VGG19(include_top=False)

if __name__ == "__main__":
    encoder.summary()
