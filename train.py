from src.train_model import build_model, X_train, y_train

def main():
    model = build_model()
    model.fit(X_train, y_train, epochs=15, batch_size=32)
    model.save("models/fake_account_detection_model.h5")
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
