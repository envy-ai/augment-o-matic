import os
import yaml
import sys
import Augmentor


def load_config(config_path):
    """
    Load transformation configuration from a YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_pipeline(input_path, output_path, config):
    """
    Set up an Augmentor pipeline using the transformations specified in the config.
    """
    pipeline = Augmentor.Pipeline(source_directory=input_path, output_directory=output_path)

    # Add transformations based on the config file
    for transformation in config['transformations']:
        name = transformation['name']
        params = transformation.get('parameters', {})
        if params['probability'] == 0:
            continue

        # Get the function from the pipeline
        function = getattr(pipeline, name, None)
        if function:
            function(**params)

    return pipeline


def main(input_path, output_path, n, config_path='config.yaml'):
    """
    Main function to generate N augmentations for each image.
    """
    # Load configuration
    config = load_config(config_path)

    # Setup pipeline
    pipeline = setup_pipeline(input_path, output_path, config)

    # Sample N augmentations for each image
    pipeline.sample(n)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python augmentomatic.py <config_yaml> <input_dir> <output_dir> <number_of_images_to_generate>")
        sys.exit(1)

    config_yaml = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    n = int(sys.argv[4])

    main(input_path, output_path, n, config_yaml)
