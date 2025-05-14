import argparse
import json
import os
import sys

# Ensure the current directory is added to sys.path
# This helps if the script is called from elsewhere but pipelines are in the root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Import Pipeline Functions --- 
# As more pipelines are added, import their main runner functions here
try:
    from sam2_eval_pipeline import run_evaluation_pipeline
except ImportError as e:
    print(f"Error importing pipeline functions: {e}")
    print("Please ensure pipeline scripts (e.g., sam2_eval_pipeline.py) exist in the project root.")
    sys.exit(1)

# --- Configuration Loading --- 
def load_config(config_path):
    """Loads JSON config, resolves relative paths to absolute, creates output dir."""
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file {config_path}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        raise

    # Resolve paths relative to the project root 
    # Important keys needing path resolution:
    path_keys = ['data_path', 'image_base_dir', 'output_path']
    for key in path_keys:
        if key in config:
            config[key] = os.path.abspath(config[key])
            # print(f"Resolved {key}: {config[key]}") # Optional: for debugging
        else:
            print(f"Warning: Expected path key '{key}' not found in config file.")

   
    if 'output_path' in config:
        output_dir = os.path.dirname(config['output_path'])
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            
    
    # Add check for pipeline_name explicitly
    if 'pipeline_name' not in config:
         print(f"Error: 'pipeline_name' key missing in configuration file: {config_path}")
         raise KeyError("'pipeline_name' not specified in config.")

    return config

# --- Pipeline Mapping --- 
# Map pipeline names (from config) to their runner functions
PIPELINE_MAP = {
    "sam2_eval": run_evaluation_pipeline,
    # Add other pipelines here as they are developed
}

# --- Main Execution Logic --- 
def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation pipelines for SAM2 analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the JSON configuration file for the experiment."
    )

    args = parser.parse_args()

    # Handle potential auth setup messages
    if not os.environ.get('HUGGING_FACE_HUB_TOKEN') and not os.path.exists(os.path.expanduser('~/.cache/huggingface/token')):
        print("Info: Hugging Face token not found in environment variables (HUGGING_FACE_HUB_TOKEN) or cache.")
        print("      Model downloads might fail or be rate-limited if required.")
        print("      Consider running 'huggingface-cli login' or setting the environment variable.")

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Failed to load or process configuration. Exiting. Error: {e}")
        sys.exit(1)

    # Get the pipeline function based on the config
    pipeline_name = config.get("pipeline_name")
    pipeline_function = PIPELINE_MAP.get(pipeline_name)

    if pipeline_function:
        print(f"--- Running Pipeline: {pipeline_name} ---")
        try:
            pipeline_function(config) # Pass the whole config dict
            print(f"--- Pipeline {pipeline_name} Finished Successfully ---")
        except Exception as e:
             print(f"--- Pipeline {pipeline_name} Failed --- ")
             print(f"Error during pipeline execution: {e}")
             
             sys.exit(1) 
    else:
        print(f"Error: Unknown pipeline name '{pipeline_name}' specified in config.")
        print(f"Available pipelines: {list(PIPELINE_MAP.keys())}")
        sys.exit(1)

if __name__ == "__main__":
    main()
