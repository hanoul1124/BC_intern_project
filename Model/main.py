import typer
from pathlib import Path
from model import training
from preprocess import preprocessing
from postprocess import postprocessing
from inference import classification


app = typer.Typer()


@app.command()
def data_preprocess(
    file: Path = typer.Argument(...),
    new_path: Path = typer.Argument(...),
):
    preprocessing(file, new_path)


@app.command()
def data_postprocess(
    file: Path = typer.Argument(...),
    new_path: Path = typer.Argument(...),
):
    postprocessing(file, new_path)


@app.command()
def train(
    train_set: Path = typer.Argument(...),
    test_set: Path = typer.Argument(...),
    checkpoint: Path = typer.Argument(None),
):
    training(train_set, test_set, checkpoint)


@app.command()
def infer(
    data: Path = typer.Argument(...),
):
    classification(data)


if __name__ == "__main__":
    app()
