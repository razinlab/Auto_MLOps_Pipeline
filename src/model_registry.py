import os
import mlflow
from mlflow.tracking import MlflowClient
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_mlflow(tracking_uri="file:./mlruns"):

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    return client

def register_model(run_id, model_name):
    client = setup_mlflow()

    try:
        client.get_registered_model(model_name)
        logger.info(f"Model {model_name} already exists in registry")
    except:
        client.create_registered_model(model_name)
        logger.info(f"Created new model {model_name} in registry")

    model_uri = f"runs:/{run_id}/model"

    model_version = mlflow.register_model(model_uri, model_name)
    logger.info(f"Registered model {model_name} version {model_version.version} from run {run_id}")

    return model_version

def get_model_versions(model_name, max_results=10):
    client = setup_mlflow()
    try:
        versions = client.search_model_versions(f"name='{model_name}'", max_results=max_results)
        return versions
    except Exception as e:
        logger.error(f"Error getting versions for {model_name}: {e}")
        return []




def set_model_alias(model_name, version, alias="champion"):
    client = setup_mlflow()
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=str(version)
        )
        logger.info(f"Set alias '{alias}' for {model_name} version {version}")
        return True
    except Exception as e:
        logger.error(f"Error setting alias '{alias}' for {model_name} v{version}: {e}")
        raise

def delete_model_alias(model_name, alias):
    client = setup_mlflow()
    try:
        client.delete_registered_model_alias(
            name=model_name,
            alias=alias
        )
        logger.info(f"Deleted alias '{alias}' from {model_name}")
        return True
    except Exception as e:
        logger.warning(f"Could not delete alias '{alias}' from {model_name}: {e}")
        return False


def promote_model_to_production(model_name, version, archive_existing=True):
    client = setup_mlflow()

    if archive_existing:
        try:
            current_champion_uri = f"models:/{model_name}@champion"
            current_model = mlflow.pyfunc.load_model(current_champion_uri)
            delete_model_alias(model_name, "champion")
            logger.info(f"Removed 'champion' alias from previous version")
        except:
            logger.info(f"No existing 'champion' alias to remove")

    set_model_alias(model_name, version, "champion")

    set_model_alias(model_name, version, "challenger")

    logger.info(f"Promoted {model_name} version {version} to champion/challenger")
    return True

def load_model(model_name, alias="champion"):
    try:
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded {model_name} model with alias '{alias}'")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def compare_models(model_name, metric_name="f1_score"):
    client = setup_mlflow()
    versions = get_model_versions(model_name)  # ✅ Use the defined function

    results = {}
    for version in versions:
        run = client.get_run(version.run_id)
        metric_value = run.data.metrics.get(metric_name)

        # Get aliases for this version instead of stage
        aliases = getattr(version, 'aliases', [])
        alias_str = f"({', '.join(aliases)})" if aliases else "(no alias)"
        results[f"Version {version.version} {alias_str}"] = metric_value

    print(f"\nModel Comparison for {model_name} based on {metric_name}:")
    for version, metric in sorted(results.items(), key=lambda x: x[1] if x[1] is not None else -float('inf'),
                                  reverse=True):
        if metric is not None:
            print(f"{version}: {metric:.4f}")
        else:
            print(f"{version}: No {metric_name} recorded")

    return results


def list_registered_models():
    client = setup_mlflow()
    try:
        models = client.search_registered_models()
        print("\nRegistered Models:")
        for model in models:
            print(f"- {model.name}")

            versions = get_model_versions(model.name, max_results=5)

            if versions:
                for version in versions:

                    aliases = getattr(version, 'aliases', [])
                    alias_str = f" [{', '.join(aliases)}]" if aliases else ""

                    try:
                        run = client.get_run(version.run_id)
                        metrics = run.data.metrics
                        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()
                                                 if k in ["accuracy", "f1_score", "sharpe_ratio"]])
                        print(f"  Version {version.version}{alias_str} ({metrics_str})")
                    except:
                        print(f"  Version {version.version}{alias_str} (no metrics)")

        return models
    except Exception as e:
        logger.error(f"Error listing registered models: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    try:
        client = setup_mlflow()

        list_registered_models()

        # Left commented out for testing
        # Register a model
        # run_id = "run_id_here"
        # model_name = "daily-price-forecaster"
        # register_model(run_id, model_name)

        # Promote a model to production
        # promote_model_to_production("daily-price-forecaster")

        # Compare models
        # compare_models("daily-price-forecaster", "sharpe_ratio")

    except Exception as e:
        logger.error(f"Error in model registry management: {e}")
