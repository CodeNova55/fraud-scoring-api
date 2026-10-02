import json

from .model import model_path, save_bundle, train_model


def main():
    bundle = train_model()
    path = model_path()
    save_bundle(bundle, path)
    metrics_text = json.dumps(bundle["metrics"], indent=2)
    path.with_name("metrics.json").write_text(metrics_text, encoding="utf-8")
    print(metrics_text)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    main()
