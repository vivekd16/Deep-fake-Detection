# Deepfake Face Detection with ResNet and Gradio

This project trains a ResNet classifier on your `real_vs_fake/real-vs-fake` dataset from a Jupyter notebook and serves the best checkpoint through a Gradio web app.

## Dataset layout

The code expects this structure, which matches your workspace:

```text
deepfakedetection/
|-- train.csv
|-- valid.csv
|-- test.csv
`-- real_vs_fake/
    `-- real-vs-fake/
        |-- train/
        |   |-- fake/
        |   `-- real/
        |-- valid/
        |   |-- fake/
        |   `-- real/
        `-- test/
            |-- fake/
            `-- real/
```

## Use your existing venv

Activate the virtual environment you already created, then run the scripts from that environment. The project code does not depend on the system `python 3.14`.

## Train the model

Training is notebook-first now. Open [deepfake_training.ipynb](/c:/Vivek/code/deepfakedetection/deepfake_training.ipynb) and run the cells.

The current default setup is a faster presentation-friendly run:

- `resnet18`
- `10` epochs
- lighter augmentation
- outputs saved to `artifacts/`

If you want to tweak the run, edit the `config = Config(...)` cell in the notebook before starting training.

Artifacts are saved into `artifacts/`:

- `best_model.pt`
- `best_valid_metrics.json`
- `test_metrics.json`
- `history.json`
- `config.json`

## View analytics

Open [descriptive_analytics.ipynb](/c:/Vivek/code/deepfakedetection/descriptive_analytics.ipynb) to view:

- dataset split and class balance
- best validation and test metrics
- confusion matrix
- training curves
- per-class metric plots

## Launch the Gradio app

After training:

```powershell
python app.py --checkpoint artifacts/best_model.pt
```

Open `http://127.0.0.1:7860` in your browser.

## Notes

- The app includes Grad-CAM overlays to show which image regions influenced the prediction.
- If you want a stronger final model later, you can raise epochs or switch back to `resnet50` in the training notebook.
- For presentation use, focus on `deepfake_training.ipynb`, `descriptive_analytics.ipynb`, `app.py`, and the files in `artifacts/`.
