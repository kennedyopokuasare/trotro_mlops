import os

from azure.ai.ml import Input, MLClient, dsl, load_component
from azure.identity import DefaultAzureCredential
from scripts.utils import register_data
import webbrowser

@dsl.pipeline(compute="serverless", description="Trotro duration prediction")
def trotro_pipeline(train_input, validate_input):

    component_dir = "./components"
    prepare_data_command_component = load_component(
        source=os.path.join(component_dir, "prepare_data.yaml")
    )

    train_command_component = load_component(
        source=os.path.join(component_dir, "train_model.yaml")
    )

    data_prep_step = prepare_data_command_component(
        train_data_path=train_input, validate_data_path=validate_input
    )

    train_command_component(
        train_data_path=data_prep_step.outputs.features_train_path,
        validation_data_path=data_prep_step.outputs.features_validation_path,
    )

    return {
        "train_data": data_prep_step.outputs.features_train_path,
        "validate_data": data_prep_step.outputs.features_validation_path,
    }


def prepare_pipeline_job():

    experiment_name = "trotr-duration-prediction"

    ml_client = MLClient.from_config(
        credential=DefaultAzureCredential(), path="./azure_ml_config.json"
    )

    train_data, validate_data, _ = register_data(ml_client=ml_client)

    pipeline = trotro_pipeline(
        train_input=Input(type="url_file", path=train_data.path),
        validate_input=Input(type="url_file", path=validate_data.path),
    )

    return ml_client.jobs.create_or_update(pipeline, experiment_name=experiment_name)


if __name__ == "__main__":

   pipeline_job = prepare_pipeline_job()
   print(f"Job started at {pipeline_job.studio_url} ")
